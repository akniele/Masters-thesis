import torch
import pandas as pd

import numpy as np
from tqdm import tqdm
from scipy.spatial import distance  # distance.cityblock gives manhattan distance
from scipy.optimize import minimize
import pickle
import sys
import config

sys.path.append('../GetClusters')
from GetClusters.differenceMetrics import sort_probs
#from ClassifierFiles.classifier import BertClassifier
from scipy.stats import entropy # kl-divergence/relative entropy if optional parameter qk is given, else calculate Shannon entropy
from Transformation.fill_up_distributions import fill_multiple_distributions


def create_prediction_data(probs1):  # held out data that has not been used to create features
    predict_data = []
    for i, sheets in enumerate(probs1[27:]):
        sequence = []
        for j, samples in enumerate(sheets):
            sorted_probs = sorted(samples, reverse=True)
            sequence.append(sorted_probs[:256])
        predict_data.append(sequence)

    predict_dict = {
      "text": predict_data}
    df = pd.DataFrame(predict_dict)
    return df


def transformProbabilities(bigprobs):
    families = classifyProbabilityIntoFamily(bigprobs)
    new_probs_sequence = []
    for i, sequence in enumerate(families):
      print(f"shape familiy: {sequence.size()}")
      for family in sequence:
        if (family == 0):
          print("0")
          # new_probs = trans0(bigprobs[i])
        elif (family == 1):
          print("1")
          # new_probs = trans1(bigprobs[i])
        else:
          print("2")
          # new_probs = trans2(bigprobs[i])
        #new_probs_sequence.append(new_probs)

    # return new_probs_sequence


# Fredrik's metric, for this purpose timeStepDiffs and sampleDiffs returns the same
def weightedManhattanDistance(dist1, dist2, probScaleLimit=0.2):
    dist1 = torch.FloatTensor(dist1)
    dist2 = torch.FloatTensor(dist2)
    probSums = dist1 + dist2
    belowThreshold = torch.where(probSums < probScaleLimit, 1, 0)
    belowThresholdMask = (belowThreshold / probScaleLimit) * probSums
    overThresholdMask = 1 - belowThreshold
    weightMask = belowThresholdMask + overThresholdMask

    absDiff = torch.abs(dist1 - dist2) * weightMask
    timeStepDiffs = torch.sum(absDiff, dim=-1)
    sampleDiffs = torch.sum(timeStepDiffs, dim=-1)
    return absDiff, timeStepDiffs, sampleDiffs


# sort prob distributions by the probs of the small distribution
# cut off after 128
# calculate distance metrics
# compare results to when not cutting off distributions

"""
    Checks if the Manhattan Distance and weighted Manhattan Distance give similar results if you only
    use the top 128 probs (ordered by the small probabilities
"""


def sanity_check_distance_metrics(new_probs, small_probs, big_probs):
    improvement_manhattan = []
    improvement_weighted_manhattan = []

    for i, sheet in tqdm(enumerate(big_probs[:2]), desc="Comparisons"):  # right now there's 100 sheets
        for j, timestep in enumerate(sheet):  # 64 time steps per sheet
            big_sorted, small_sorted = sort_probs(big_probs[i][j], small_probs[i][j])
            new_sorted, _ = sort_probs(new_probs[i][j], small_probs[i][j])

            big_sorted = big_sorted[:128]
            small_sorted = small_sorted[:128]
            new_sorted = new_sorted[:128]

            diff_changed = distance.cityblock(small_sorted, new_sorted)  # calculates Manhattan distance
            diff_unchanged = distance.cityblock(small_sorted, big_sorted)

            _, weighted_diff_changed, _ = weightedManhattanDistance(small_sorted, new_sorted)
            _, weighted_diff_unchanged, _ = weightedManhattanDistance(small_sorted, big_sorted)
            # for now, if diff_changed is smaller than diff_unchanged, we'll count that as a success
            # later on: use black-box model as baseline, the closer diff_changed to black-box model's diff, the better
            if diff_changed < diff_unchanged:
                improvement_manhattan.append(True)
            else:
                improvement_manhattan.append(False)

            if weighted_diff_changed < weighted_diff_unchanged:
                improvement_weighted_manhattan.append(True)
            else:
                improvement_weighted_manhattan.append(False)

    improvement_score_manhattan = sum(improvement_manhattan) / len(improvement_manhattan)
    improvement_score_weighted_manhattan = sum(improvement_weighted_manhattan) / len(improvement_weighted_manhattan)

    return improvement_score_manhattan, improvement_score_weighted_manhattan


"""
    Returns the percentage of new distributions that are closer to the small distribution after
    the transformations
    
"""


def compare_distributions(new_probs, small_probs, big_probs):
    improvement_manhattan = []
    improvement_weighted_manhattan = []
    print(f"shape big probs: {big_probs.shape}")
    for i, sheet in tqdm(enumerate(big_probs), desc="Comparisons"):  # right now there's 100 sheets
        for j, timestep in enumerate(sheet):  # 64 time steps per sheet
            diff_changed = distance.cityblock(small_probs[i][j], new_probs[i][j])  # calculates Manhattan distance
            diff_unchanged = distance.cityblock(small_probs[i][j], big_probs[i][j])

            _, weighted_diff_changed, _ = weightedManhattanDistance(small_probs[i][j], new_probs[i][j])
            _, weighted_diff_unchanged, _ = weightedManhattanDistance(small_probs[i][j], big_probs[i][j])
            # for now, if diff_changed is smaller than diff_unchanged, we'll count that as a success
            # later on: use black-box model as baseline, the closer diff_changed to black-box model's diff, the better
            if diff_changed < diff_unchanged:
                improvement_manhattan.append(True)
            else:
                improvement_manhattan.append(False)

            if weighted_diff_changed < weighted_diff_unchanged:
                improvement_weighted_manhattan.append(True)
            else:
                improvement_weighted_manhattan.append(False)

    #improvement_score_manhattan = sum(improvement_manhattan) / len(improvement_manhattan)
    improvement_score_weighted_manhattan = sum(improvement_weighted_manhattan) / len(improvement_weighted_manhattan)

    return improvement_score_weighted_manhattan


"""
    Returns a list of the Manhattan Distance between each pair of distributions, and another one
    with a list of the weightedManhattan Distance for each pair of distributions

"""


def compare_distance(new_probs, small_probs, big_probs):
    distances_manhattan_changed = []
    distances_manhattan_unchanged = []
    distances_weighted_changed = []
    distances_weighted_unchanged = []

    for i, sheet in tqdm(enumerate(big_probs), desc="Comparisons"):  # right now there's 100 sheets
        for j, timestep in enumerate(sheet):  # 64 time steps per sheet
            diff_changed = distance.cityblock(small_probs[i][j], new_probs[i][j])  # calculates Manhattan distance
            diff_unchanged = distance.cityblock(small_probs[i][j], big_probs[i][j])

            _, weighted_diff_changed, _ = weightedManhattanDistance(small_probs[i][j], new_probs[i][j])
            _, weighted_diff_unchanged, _ = weightedManhattanDistance(small_probs[i][j], big_probs[i][j])
            # for now, if diff_changed is smaller than diff_unchanged, we'll count that as a success
            # later on: use black-box model as baseline, the closer diff_changed to black-box model's diff, the better
            distances_manhattan_changed.append(diff_changed)
            distances_manhattan_unchanged.append(diff_unchanged)

            distances_weighted_changed.append(weighted_diff_changed)
            distances_weighted_unchanged.append(weighted_diff_unchanged)

            total_weighted_distance_changed = distances_weighted_changed  # sum(distances_weighted_changed)
            total_weighted_distance_unchanged = distances_weighted_unchanged  # sum(distances_weighted_unchanged)

    return total_weighted_distance_changed, total_weighted_distance_unchanged


"""
  1. sort the big probability distribution descendingly, save indices
  2. for each bucket, change the total probability by as much as the difference metric suggested would be good
  --> take some kind of an average for all of the distributions in the same cluster?
  3. unsort the resulting distribution based on the indices saved in step 1
  4. return the distribution

"""


def not_used_trans0(bigprobs):
    number_of_buckets = 3
    indices = [0, 10, 35, len(bigprobs)]
    bucket_probs = [-0.167, 0.0097, 0.1573]

    probs = bigprobs.copy()
    probs = torch.from_numpy(probs)  # needs to be a tensor for using torch.sort

    # sort probabilities, get sorted probabilities and indices
    sorted_probs, sorted_indices = torch.sort(probs, dim=0, descending=True)
    sorted_probs, sorted_indices = sorted_probs.numpy(), sorted_indices.numpy()

    leftover_prob = [0 for i in range(number_of_buckets)]

    for i in range(number_of_buckets):  # loop through the buckets
        bucket_before = sum(sorted_probs[indices[i]:indices[i + 1]])  # total probability bucket has

        if bucket_before < -(bucket_probs[i]):  # if p(bucket x) < bucket_probs[x]
          # don't substract full bucket_prob from this bucket, set probs in bucket to 0 instead
          # figure out how much probability is not accounted for then
            leftover_prob[i] = bucket_probs[
                             i] + bucket_before  # how much probability needs to be subtracted from other buckets
            sorted_probs[indices[i]:indices[i + 1]] = sorted_probs[indices[i]:indices[i + 1]] * 0
            continue

        bucket_after = sum(sorted_probs[indices[i]:indices[i + 1]]) + bucket_probs[
        i]  # total probability we want bucket to have

        sorted_probs[indices[i]:indices[i + 1]] = sorted_probs[indices[i]:indices[
        i + 1]] / bucket_before * bucket_after  # assign new probs to elements in bucket

    num_buckets_to_modify = leftover_prob.count(0)

    if num_buckets_to_modify == number_of_buckets:
        pass
    else:
        modification_per_bucket = sum(leftover_prob) / num_buckets_to_modify

        done = False
        while not done:
            for i, prob in enumerate(leftover_prob):
                if prob == 0:
                    if sum(sorted_probs[indices[i]:indices[i + 1]]) > -modification_per_bucket:
                        bucket_before = sum(sorted_probs[indices[i]:indices[i + 1]])
                        bucket_after = sum(sorted_probs[indices[i]:indices[i + 1]]) + modification_per_bucket
                        sorted_probs[indices[i]:indices[i + 1]] = sorted_probs[
                                                          indices[i]:indices[i + 1]] / bucket_before * bucket_after
                        if i == len(leftover_prob) - 1:
                            done = True
                            break

                    else:
                        bucket_before = sum(sorted_probs[indices[i]:indices[i + 1]])
                        leftover_prob[i] += (bucket_before + modification_per_bucket)
                        num_buckets_to_modify = leftover_prob.count(0)
                        modification_per_bucket = sum(leftover_prob) / num_buckets_to_modify
                        sorted_probs[indices[i]:indices[i + 1]] = sorted_probs[indices[i]:indices[i + 1]] * 0
                        break

    final_probs = np.zeros(bigprobs.shape)
    for i, index in enumerate(sorted_indices):
        final_probs[index] = sorted_probs[i]

    return final_probs


def f(beta, p, entropy_small):  # solution found here: https://stats.stackexchange.com/questions/521582/controlling-the-entropy-of-a-distribution
    z = sum(p**beta)
    new_entropy = (-1 / z) * sum((p**beta) * (beta * np.log(p) - np.log(z)))
    return (new_entropy - entropy_small)**2


def trans_1(bigprobs, mean_entropy):
    """
    :param bigprobs: a single probability distribution from the big model
    :param smallprobs: a single probability distribution from the small model
    :param mean_entropy: the mean entropy of all distributions from the small model (from training data)
    :return: the probability distribution from the big model, transformed to approximate the entropy of the small model
    """
    # change the entropy of the big probability distribution to make it more similar to the entropy of the smaller model
    p = bigprobs
    p = np.expand_dims(p, 1)  # unsqueeze p (optimizer wants (n,1))

    small_entropy = mean_entropy
    #4.361379
    #entropy(smallprobs)

    solution = minimize(fun=f, x0=1, args=(p, small_entropy)) # find minimum of function f, initial guess is set to 1 because prob**1 is just prob
    new_z = sum(p**solution.x)
    transformed_p = (p**solution.x) / new_z
    transformed_p = np.squeeze(transformed_p, 1)  # squeeze away the dimension we needed for the optimizer
    return transformed_p  # return the big model's probs, transformed to have the same entropy as the small model's probs


def trans_0(bigprobs, mean_bucket_trans, bucket_indices):

    sorted_indices = bigprobs.argsort()[:, :, ::-1]

    depth = np.arange(len(bigprobs))
    depth = np.expand_dims(depth, 1)
    depth = np.expand_dims(depth, 2)
    depth = np.broadcast_to(depth, bigprobs.shape)

    rows = np.arange(bigprobs.shape[1])
    rows = np.expand_dims(rows, 1)
    rows = np.broadcast_to(rows, bigprobs.shape)

    sorted_big_probs = bigprobs[depth, rows, sorted_indices]

    del bigprobs

    current_bucket_probs = np.ones((sorted_big_probs.shape[0], sorted_big_probs.shape[1], len(bucket_indices)-1))

    for i, index in enumerate(bucket_indices[:-1]):  # get current bucket probabilities
        current_bucket_probs[:, :, i] = np.sum(sorted_big_probs[:, :, index:bucket_indices[i + 1]], -1)

    new_bucket_trans = current_bucket_probs + mean_bucket_trans  # add up current bucket probs and transformations from train data

    min_cols = np.amin(new_bucket_trans, axis=-1)  # get min bucket prob

    output = min_cols < 0  # create boolean mask, True if there's a bucket prob < 0, else False

    def add_min_and_normalize(x):  # called if there's a bucket prob < 0, adds this prob to all probs, then normalizes
        x2 = x + np.expand_dims(abs(np.min(x, axis=-1)), -1)
        return x2 / np.expand_dims(np.sum(x2, axis=-1), -1)

    def normalize(x):  # normalizes the probabilities (squeezes them between 0 and 1)
        return x / np.expand_dims(np.sum(x, axis=-1), -1)

    new_bucket_trans[output, :] = add_min_and_normalize(new_bucket_trans[output, :])  # apply add_min_and_normalize to rows with bucket prob < 0
    new_bucket_trans[~output, :] = normalize(new_bucket_trans[~output, :])  # apply normalize function to rows without bucket prob < 0

    target_transformation = new_bucket_trans - current_bucket_probs # get final bucket transformation, i.e. how much to add / subtract from each bucket after normalization

    for i, index in enumerate(bucket_indices[:-1]):  # get current bucket probabilities
        sorted_big_probs[:, :, index:bucket_indices[i+1]] = sorted_big_probs[:, :, index:bucket_indices[i + 1]] + \
                                                     np.expand_dims(target_transformation[:, :, i] /
                                                                    (bucket_indices[i + 1] - index), -1)

    final_probs = sorted_big_probs[depth, rows, sorted_indices]  # unsort the probabilities

    return final_probs


def transformations(bigprobs, indices, mean_features, bucket_indices, functions, pred_labels=None):
    """
    1. fill up distributions
    2. create boolean array with pred labels ->
    3. for each distribution of a certain label, transform distribution using the relevant mean features
    4. return array with the transformed distributions
    :param bigprobs:
    :param pred_labels:
    :param mean_features: dictionary with mean features
    :return:
    """
    print(f"entropy big probs first distribution: {entropy(bigprobs[0][0], axis=-1)}")

    print("  => FILL UP DISTRIBUTIONS")
    transformed_probs = fill_multiple_distributions(bigprobs, indices)

    print("  => DONE FILLING UP DISTRIBUTIONS")

    if pred_labels is not None:
        unique_labels = np.unique(pred_labels)

        num_features = [config.function_feature_dict[f"{function.__name__}"] for function in functions]
        print(f"number of features: {num_features}")

        bucket_indices.insert(0, 0)
        bucket_indices.append(16384)

        for label in tqdm(unique_labels, desc="iterating over labels"):
            means = []
            for j in range(len(num_features)):
                means.extend([mean_features[f"{functions[j].__name__}_{i}_{label}"] for i in range(num_features[j])])
            print(f"means for label {label}: {means}")

            print(f"  => FIRST TRANSFORMATION")

            print(f"means[num_features[0]:]: {means[num_features[0]:]}")

            transformed_probs[pred_labels == label] = np.apply_along_axis(trans_1, -1,
                                                                          transformed_probs[pred_labels == label],
                                                                          means[num_features[0]:])
            print(f"  => SECOND TRANSFORMATION")

            transformed_probs[pred_labels == label] = trans_0(transformed_probs[pred_labels == label],
                                                              means[:num_features[0]], bucket_indices)
            print(f"  => DONE!")

            print(f"entropy big probs first distribution: {entropy(transformed_probs[0][0], axis=-1)}")

    return transformed_probs


def evaluate_transformations(transformed_probs, bigprobs, smallprobs):
    """
    1. Compare the transformed probs to the small probs (using the average of the weighted Manhattan distance)
    2. Compare the original big probs to the small probs (using the average of the weighted Manhattan distance)
    3. See if distance between transformed and small probs is smaller than between original big probs and small probs
    4. return mean distance between trans and small probs minus mean distance between big and small probs
    :param transformed_probs:
    :param bigprobs:
    :param smallprobs:
    :return:
    """
    _, dist_trans, _ = weightedManhattanDistance(transformed_probs, smallprobs, probScaleLimit=0.2)
    _, dist_big, _ = weightedManhattanDistance(bigprobs, smallprobs, probScaleLimit=0.2)

    print(f"shape dist_trans: {dist_trans.shape}")
    print(f"shape dist_big: {dist_big.shape}")

    mean_dist_trans = np.mean(dist_trans, axis=-1)
    mean_dist_big = np.mean(dist_big, axis=-1)

    print(f" mean_dist_trans: {mean_dist_trans}")
    print(f" mean_dist_big: {mean_dist_big}")

    return mean_dist_trans - mean_dist_big


if __name__ == "__main__":
    # f = lambda x: torch.tensor(x, dtype=torch.float32)
    #
    # unchanged = f([0.3, 0.2, 0.5])
    # print("Unchanged", unchanged)
    #
    # transformation = f([-.2, 0.3, -0.3])
    # print("Transformation", transformation)
    #
    # actual_trans = trans_0(unchanged, transformation, indices=[1, 4])

    # arr = np.array([[[0.3, 0.3, 0.4],
    #                  [0.3, 0.9, 0.4],
    #                  [0.1, 0.7, 0.4]],
    #
    #                 [[0.3, 0.2, 0.9],
    #                  [0.1, 0.1, 0.8],
    #                  [0.3, 0.2, 0.5]]])
    #
    # mean_bucket_trans = np.array([-0.2, 0.4])
    # mean_bucket_trans = np.expand_dims(mean_bucket_trans, 0)
    # mean_bucket_trans = np.expand_dims(mean_bucket_trans, 0)
    #
    # indices = [0, 1, 3]
    #
    # target_probs = trans_0(arr, mean_bucket_trans, indices)
    #
    # print(target_probs)


    # NUM_TRAIN_SHEETS = 20
    # NUM_TEST_SHEETS = 12
    #
    """Load data"""
    # data
    probs1 = pickle.load(open("../train_data/train_big_100000_0.pkl", "rb"))
    indices = pickle.load(open("../train_data/indices_big_100000_0.pkl", "rb"))
    probs1 = probs1.detach().numpy()
    indices = indices.detach().numpy()

    print(f"shape probs: {probs1.shape}")

    probs1 = fill_multiple_distributions(probs1[:500], indices[:500], topk=256)

    mean_bucket_trans = np.array([[[-0.2, 0.5, 0.4]]])

    bucket_indices = [0, 20, 50, probs1.shape[-1]]

    trans_probs = trans_0(probs1, mean_bucket_trans, bucket_indices)

    print(trans_probs.shape)
    print(trans_probs[:1, :, :])

    # print(f"entropy small model: {entropy(probs0[0][0])}")
    # print(f"entropy big model: {entropy(probs1[0][0])}")
    # trans_p = trans1(probs1[0][0], probs0[0][0], 0.0210886001586914)
    # print(entropy(trans_p))



    #
    # """transform big probs to be more similar to small probs"""
    #
    # transformed_probs = probability_transformation(probs1[:20, :, :], probs0[:20, :, :])
    # print(transformed_probs.shape)
    #
    # total_weighted_changed, total_weighted_unchanged = compare_distance(transformed_probs, probs0[:20, :, :], probs1[:20, :, :])
    #
    # print(f"standard deviation of weighted Manhattan distances after transformation: {np.std(total_weighted_changed)}")
    # print(f"standard deviation of weighted Manhattan distances without transformation: {np.std(total_weighted_unchanged)}")
    #
    # print(f"mean of weighted Manhattan distances after transformation: {np.mean(total_weighted_changed)}")
    # print(
    #     f"mean of weighted Manhattan distances without transformation: {np.mean(total_weighted_unchanged)}")
    #
    #
    # # print(f"sum of Manhattan Distance after transformation: {total_changed}")
    # # print(f"sum of Manhattan Distance without transformation: {total_unchanged}")
    #
    # print(f"sum of weighted Manhattan Distance after transformation: {sum(total_weighted_changed)}")
    # print(f"sum of weighted Manhattan Distance without: {sum(total_weighted_unchanged)}")
    #
    # #transformed_probs = probability_transformation(probs1, probs0)
    #
    # improvement_weighted = compare_distributions(transformed_probs, probs0[:20, :, :], probs1[:20, :, :])
    #
    # print(f" Using the Manhattan distance, % "  #{improvement_manhattan * 100}
    #       f"of the transformed distributions are closer to the target distribution.\nFor the weighted Manhattan "
    #       f"distance, the score is {improvement_weighted * 100}%.")
    #
    # improvement_manhattan_128, improvement_weighted_128 = sanity_check_distance_metrics(transformed_probs,
    #                                                                                     probs0[:20, :, :], probs1[:20, :, :])
    # print(f" Using the cutoff Manhattan distance, {improvement_manhattan_128 * 100}% "
    #       f"of the transformed distributions are closer to the target distribution.\nFor the cutoff weighted Manhattan "
    #       f"distance, the score is {improvement_weighted_128 * 100}%.")
