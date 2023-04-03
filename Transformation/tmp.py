import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy
import torch
import pickle

# f = lambda x: torch.tensor(x, dtype=torch.float32)
#
#
# def normalize(x, dim=0):
#     if(x.min() < 0):
#         x2 = x + abs(x.min())
#         return x2 / x2.sum()
#     return x / x.sum()
#
#
# unchanged = f([0.3, 0.2, 0.5])
# #print(normalize(unchanged))
# print("Unchanged", unchanged)
# transformation = f([-.4, 0.3, -0.6])
# print("Transformation", transformation)
# changedTransformation = unchanged + transformation
# print("Changed Trans:", changedTransformation)
# finalDistribution = normalize(changedTransformation, dim=0)
# print("Final", finalDistribution)
#
# actualTransformation = finalDistribution - unchanged
# print("Actual Trans", actualTransformation)

# [0.1, 0.1, 0.1] -> [0.1 - 0.036, 0.1 - 0.036, 0.1 - 0.036]


# def weightedManhattanDistance(dist1, dist2, probScaleLimit=0.3):
#     dist1 = torch.FloatTensor(dist1)
#     dist2 = torch.FloatTensor(dist2)
#     probSums = dist1 + dist2
#     print(f"sum probs: {probSums}")
#     belowThreshold = torch.where(probSums < probScaleLimit, 1, 0)
#     print(f"below Threshold: {belowThreshold}")
#     belowThresholdMask = (belowThreshold / probScaleLimit) * probSums
#     print(f"belowThresholdMask: {belowThresholdMask}")
#     overThresholdMask = 1 - belowThreshold
#     print(f"overThresholdMask: {overThresholdMask}")
#     weightMask = belowThresholdMask + overThresholdMask
#     print(f"weightMask: {weightMask}")
#
#     absDiff = torch.abs(dist1 - dist2) * weightMask
#     print(f"absDiff: {absDiff}")
#     timeStepDiffs = torch.sum(absDiff, dim=-1)
#     print(f"timeStepDiffs: {timeStepDiffs}")
#     sampleDiffs = torch.sum(timeStepDiffs, dim=-1)
#     print(f"sampleDiffs: {sampleDiffs}")
#     return absDiff, timeStepDiffs, sampleDiffs

def f(beta, p, entropy_small):  # solution found here: https://stats.stackexchange.com/questions/521582/controlling-the-entropy-of-a-distribution
    z = sum(p**beta)
    new_entropy = (-1 / z) * sum((p**beta) * (beta * np.log(p) - np.log(z)))
    return 1 # (new_entropy - entropy_small)**2


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

    #b = (1e-8, None)

    solution = minimize(fun=f, x0=1, args=(p, small_entropy)) # find minimum of function f, initial guess is set to 1 because prob**1 is just prob
    print(f"solution: {solution.x}")
    new_z = sum(p**solution.x)
    print(f"new_z: {new_z}")
    transformed_p = (p**solution.x) / new_z
    transformed_p = np.squeeze(transformed_p, 1)  # squeeze away the dimension we needed for the optimizer
    return transformed_p  # return the big model's probs, transformed to have the same entropy as the small model's probs


if __name__ == "__main__":

    with open(f"../train_data/train_big_100000_9.pkl", "rb") as f:
        bigprobs = pickle.load(f)

    bigprobs = bigprobs[:100].numpy()

    entropy_bigprobs = entropy(bigprobs, -1)

    print(np.min(entropy_bigprobs))

    # check entropy function ---------------------------------------------------------------------------------------#
    # arr = np.random.randint(1, 1000, (2, 2, 3))
    # arr = arr / np.sum(arr)
    # print(f"arr: {arr}")
    # arr2 = np.random.randint(1, 1000, (2, 2, 3))
    # arr2 = arr2 / np.sum(arr2)
    # print(f"arr2: {arr2}")
    #
    # entropy_arr = entropy(arr, axis=-1)
    # print(f"original entropy: {entropy_arr}")
    #
    # mean_entropy = entropy(arr2, axis=-1)
    # print(f"mean entropy: {mean_entropy}")
    #
    # transformed_arr = np.apply_along_axis(trans_1, -1, arr, mean_entropy[0][0])
    #
    # #transformed_arr = trans_1(arr, mean_entropy)
    #
    # #print(f"transformed_arr {transformed_arr}")
    # print(f"entropy transformed arr: {entropy(transformed_arr, axis=-1)}")

    ## --------------------------------------------------------------------------------------------------------------##

    # distr1 = np.random.default_rng().uniform(0, 0.3, (2, 2, 3))
    # print(f"distr1: {distr1}")
    # distr2 = np.random.default_rng().uniform(0, 0.3, (2, 2, 3))
    # print(f"distr2: {distr2}")
    # _, mandist, _ = weightedManhattanDistance(distr1, distr2)
    #print(f"weighted Manhattan distance: {mandist}")



