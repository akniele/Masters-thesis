import numpy as np
import torch

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


def weightedManhattanDistance(dist1, dist2, probScaleLimit=0.3):
    dist1 = torch.FloatTensor(dist1)
    dist2 = torch.FloatTensor(dist2)
    probSums = dist1 + dist2
    print(f"sum probs: {probSums}")
    belowThreshold = torch.where(probSums < probScaleLimit, 1, 0)
    print(f"below Threshold: {belowThreshold}")
    belowThresholdMask = (belowThreshold / probScaleLimit) * probSums
    print(f"belowThresholdMask: {belowThresholdMask}")
    overThresholdMask = 1 - belowThreshold
    print(f"overThresholdMask: {overThresholdMask}")
    weightMask = belowThresholdMask + overThresholdMask
    print(f"weightMask: {weightMask}")

    absDiff = torch.abs(dist1 - dist2) * weightMask
    print(f"absDiff: {absDiff}")
    timeStepDiffs = torch.sum(absDiff, dim=-1)
    print(f"timeStepDiffs: {timeStepDiffs}")
    sampleDiffs = torch.sum(timeStepDiffs, dim=-1)
    print(f"sampleDiffs: {sampleDiffs}")
    return absDiff, timeStepDiffs, sampleDiffs


if __name__ == "__main__":
    distr1 = np.random.default_rng().uniform(0, 0.3, (2, 2, 3))
    print(f"distr1: {distr1}")
    distr2 = np.random.default_rng().uniform(0, 0.3, (2, 2, 3))
    print(f"distr2: {distr2}")
    _, mandist, _ = weightedManhattanDistance(distr1, distr2)
    #print(f"weighted Manhattan distance: {mandist}")



