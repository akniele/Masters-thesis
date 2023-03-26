import numpy as np
import torch

f = lambda x: torch.tensor(x, dtype=torch.float32)


def normalize(x, dim=0):
    if(x.min() < 0):
        x2 = x + abs(x.min())
        return x2 / x2.sum()
    return x / x.sum()


unchanged = f([0.3, 0.2, 0.5])
#print(normalize(unchanged))
print("Unchanged", unchanged)
transformation = f([-.4, 0.3, -0.6])
print("Transformation", transformation)
changedTransformation = unchanged + transformation
print("Changed Trans:", changedTransformation)
finalDistribution = normalize(changedTransformation, dim=0)
print("Final", finalDistribution)

actualTransformation = finalDistribution - unchanged
print("Actual Trans", actualTransformation)

# [0.1, 0.1, 0.1] -> [0.1 - 0.036, 0.1 - 0.036, 0.1 - 0.036]

