"""
Different metrics for comparing probability distributions:
- difference in entropy (NOT relative entropy)
- difference in probability per bucket
- is the correct token in topk?
- average word length of topk tokens
- difference in the number of tokens per bucket
- combined percentage of the topk tokens

"""
import numpy as np
import math
from scipy.stats import entropy # kl-divergence/relative entropy if optional parameter qk is given, else calculate Shannon entropy


"""Difference in entropy"""


def entropy_difference(probs0, probs1):
    small_entropy = entropy(probs0)
    big_entropy = entropy(probs1)
    diff = big_entropy - small_entropy
    return [diff]


"""Difference in probability per bucket"""


def sort_probs(big_probs, small_probs, sort_by=0):  # sorts the probabilities by the big probabilities
    small_sample = small_probs.copy()  # create copies of the distributions
    big_sample = big_probs.copy()

    zipped_samples = [(big_samp, small_samp) for big_samp, small_samp in
                      zip(big_sample, small_sample)]  # zip the two prob distributions
    zipped_samples.sort(key=lambda a: a[sort_by], reverse=True)  # sort by the big model's probabilities
    big_sorted = np.array([prob1 for (prob1, prob2) in zipped_samples])
    small_sorted = np.array([prob2 for (prob1, prob2) in zipped_samples])

    return big_sorted, small_sorted


def bucket_diff_top_k(probs0, probs1, number_of_buckets=None, indices=None):
    if indices is None:
        indices = [0, 10, 35]
    if number_of_buckets is None:
        number_of_buckets = 3
    big_sample_sorted, small_sample_sorted = sort_probs(probs1, probs0)  # call function sort_probs

    indices.append(len(big_sample_sorted))  # add index of len of one sheet

    bucket_diff = []
    for i in range(number_of_buckets):
        bucket_prob_big = sum(big_sample_sorted[indices[i]:indices[i + 1]])  # total probability of big bucket
        # print(bucket_prob_big)
        bucket_prob_small = sum(small_sample_sorted[indices[i]:indices[i + 1]])  # total probability of small bucket
        # print(bucket_prob_small)
        difference = bucket_prob_big - bucket_prob_small  # absolute diff betweeen them
        bucket_diff.append(difference)

    return bucket_diff  # returns the probability of buckets of specified size k each


"""Is the correct token in the top k?"""


def get_topk_success(probs, k):  # how high is the combined percentage of the top 10 tokens?
    # And is the correct token one of the top 10?
    probs = [(item, idx) for idx, item in enumerate(probs)]  # add indices so we still know them after sorting
    sorted_probs = sorted(probs, reverse=True)  # sort list in descending order
    topk = sorted_probs[:k]  # take top 10 tokens
    success = True in list(map(lambda x: x[1] == 0, topk))  # check if one of the top 10 tokens has index 0

    if success:
        return 1

    return 0


def compare_topk_success(probs0, probs1, k=3):
    small_success = get_topk_success(probs0, k)
    big_success = get_topk_success(probs1, k)

    if small_success and big_success:
        return [0]
    elif small_success and not big_success:
        return [1]
    elif not small_success and big_success:
        return [2]
    elif not small_success and not big_success:
        return [3]
    else:
        print("You missed something!")


"""## What is the combined percentage of top k tokens?"""


def get_topk_prob(probs, k):
    sorted_probs = sorted(probs, reverse=True)
    topk = sorted_probs[:k]
    top_perc = sum(topk)

    if top_perc > 0.8:
        return 0
    elif top_perc > 0.5:
        return 1
    elif top_perc > 0.3:
        return 2
    else:
        return 3


def compare_topk_prob(probs0, probs1, k=10):
    small_probs = get_topk_prob(probs0, k)
    big_probs = get_topk_prob(probs1, k)

    if small_probs != big_probs:
        return [0]
    elif small_probs == big_probs:
        return [1]
    else:
        print("You missed something!")


"""Average word length of top k tokens"""


def average_len_topk(probs, tokens, topk):
    prob_word = [(prob, word) for prob, word in zip(probs, tokens)]
    prob_word.sort(key=lambda x: x[0], reverse=True)
    top = prob_word[:topk]
    average = sum(len(token[1][0]) for token in top) / topk
    return average


def compare_average_topk_len(probs0, probs1, tokens, topk=10):
    small_average, big_average = average_len_topk(probs0, tokens, topk), average_len_topk(probs1, tokens, topk)
    if math.isclose(big_average, small_average, rel_tol=0.0, abs_tol=0.5):
        return [1]
    else:
        return [0]


"""Difference in the number of tokens per bucket"""


def get_bucket_size(sample, number_of_buckets, bucket_probs):
    bucket = [0]
    for i in range(number_of_buckets):  # for every bucket
        total_prob = 0
        bucket_size = 0
        for samp in sample[sum(bucket[:i + 1]):]:
            if total_prob < bucket_probs[i]:
                bucket_size += 1
                total_prob += samp
            else:
                break

        bucket.append(bucket_size)

    del bucket[0]

    return bucket


def bucket_diff_top_p(probs0, probs1, number_of_buckets=3, bucket_probs=None):
    if bucket_probs is None:
        bucket_probs = [0.4, 0.1, 0.5]
    big_sample_sorted, small_sample_sorted = sort_probs(probs1, probs0)  # call function sort_probs

    bucket_small = get_bucket_size(small_sample_sorted, number_of_buckets, bucket_probs)
    bucket_big = get_bucket_size(big_sample_sorted, number_of_buckets, bucket_probs)

    difference = np.array(bucket_big) - np.array(bucket_small)  # diff between them

    return np.ndarray.tolist(difference)  # returns number of elements in buckets of specified size p