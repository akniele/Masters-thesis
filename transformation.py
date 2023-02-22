import torch
import pandas as pd
from classifier import BertClassifier
import numpy as np
from scipy.spatial import distance  # distance.cityblock gives manhattan distance


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


def classifyProbabilityIntoFamily(largeModelProbs):
    model = BertClassifier()
    model.load_state_dict(torch.load(f'first_try.pt'))
    model.eval()

    predictions = model.predict(largeModelProbs)

    return predictions  # return a number that corresponds to the family (as defined by the clusters)


def transformProbabilities(largeModelProbs):
    families = classifyProbabilityIntoFamily(largeModelProbs)
    new_probs_sequence = []
    for i, sequence in enumerate(families):
      print(f"shape familiy: {sequence.size()}")
      for family in sequence:
        if (family == 0):
          print("0")
          # new_probs = trans0(largeModelProbs[i])
        elif (family == 1):
          print("1")
          # new_probs = trans1(largeModelProbs[i])
        else:
          print("2")
          # new_probs = trans2(largeModelProbs[i])
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


def compareDistributions(newProbs, small_probs, big_probs ):
    diff_changed = distance.cityblock(small_probs, newProbs)  # calculates Manhattan distance between the distributions
    diff_unchanged = distance.cityblock(small_probs, big_probs)

    _, weighted_diff_changed, _ = weightedManhattanDistance(small_probs, newProbs)
    _, weighted_diff_unchanged, _ = weightedManhattanDistance(small_probs, big_probs)
    # for now, if diff_changed is smaller than diff_unchanged, we'll count that as a success
    # later on: use the black-box model as baseline, the closer diff_changed to the black-box models diff, the better
    if diff_changed < diff_unchanged:
        print("success!")
    else:
        print("this didn't work.")
    if weighted_diff_changed < weighted_diff_unchanged:
        print("more success!")
    else:
        print("this didn't work either.")


"""
  1. sort the big probability distribution descendingly, save indices
  2. for each bucket, change the total probability by as much as the difference metric suggested would be good
  --> take some kind of an average for all of the distributions in the same cluster?
  3. unsort the resulting distribution based on the indices saved in step 1
  4. return the distribution

"""


def trans0(largeModelProbs):
    number_of_buckets = 3
    indices = [0, 2, 3, len(largeModelProbs)]
    bucket_probs = [-0.5, 0.5, 0]

    probs = largeModelProbs.copy()
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

    final_probs = np.zeros(largeModelProbs.shape)
    for i, index in enumerate(sorted_indices):
        final_probs[index] = sorted_probs[i]

    return final_probs


def trans1(largeModelProbs):
    # change the entropy of the big probability distribution to make it more similar to the entropy of the smaller model
    pass
