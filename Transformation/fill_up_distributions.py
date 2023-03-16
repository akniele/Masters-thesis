# calculate probability mass of topk
# 1 minus that is what we need
# calculate (1-probability mass of topk) / vocab_length - topk

# create array of zeros of length vocab_length

# add the probabilities of the topk, using their original indices

# for all other indices, add the prob that you calculated earlier
import torch

import config

import numpy as np
VOCAB_LENGTH = 15


def fill_distribution(distribution, indices, topk=256):
    prob_mass = sum(distribution)
    mass_missing = 1 - prob_mass

    indices = indices.numpy().astype(int)

    prob_to_add = mass_missing / (VOCAB_LENGTH - topk)
    new_distribution = np.zeros(VOCAB_LENGTH)

    np.add.at(new_distribution, indices, distribution)
    new_distribution[new_distribution == 0] = prob_to_add

    # zero_indices = np.where(new_distribution == 0)[0]
    # np.add.at(new_distribution, zero_indices, prob_to_add)

    return new_distribution


def fill_multiple_distribution(distributions, indices, topk=10):
    #new_distribution = np.zeros((distributions.shape[0], distributions.shape[1], VOCAB_LENGTH))

    distributions = np.pad(distributions, ((0, 0), (0, 0), (0, VOCAB_LENGTH-topk)), 'constant')
    print(f"distributions after padding: {distributions}")

    distributions = torch.from_numpy(distributions)
    print(f"distributions after turning into tensor: {distributions}")

    unsorted = distributions.gather(2, indices.argsort(2))  # this removes the padding --> find solution!
    print(f"after unsorting: {unsorted}")
    new_distribution = unsorted.numpy()
    prob_mass = np.sum(new_distribution, axis=-1)
    mass_missing = 1 - prob_mass
    prob_to_add = mass_missing / (VOCAB_LENGTH - topk)
    print(f"prob to add: {prob_to_add}")
    prob_to_add = np.expand_dims(prob_to_add, -1)
    prob_to_add = np.broadcast_to(prob_to_add, new_distribution.shape)
    new_distribution[new_distribution == 0] = prob_to_add[new_distribution == 0]

    return new_distribution


if __name__ == "__main__":

    a = np.random.dirichlet(np.ones(10) * 1000., size=(2, 2))

    print(f" shape a: {a.shape}")
    print(a)

    indices = torch.tensor([[[3, 2, 0, 1, 13, 4, 14, 9, 5, 8], [0, 8, 13, 12, 3, 1, 4, 6, 2, 10]],
                            [[2, 1, 4, 10, 9, 3, 0, 11, 5, 6], [3, 0, 5, 1, 12, 14, 4, 6, 13, 2]]])

    print(f"indices shape: {indices.shape}")

    new_array = fill_multiple_distribution(a, indices, topk=10)


    print(new_array)
    print(np.sum(new_array, axis=-1))






