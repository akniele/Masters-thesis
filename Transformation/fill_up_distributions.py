# calculate probability mass of topk
# 1 minus that is what we need
# calculate (1-probability mass of topk) / vocab_length - topk

# create array of zeros of length vocab_length

# add the probabilities of the topk, using their original indices

# for all other indices, add the prob that you calculated earlier

import numpy as np
VOCAB_LENGTH = 16384


def fill_distribution(distribution, indices, topk=256):
    prob_mass = sum(distribution)
    mass_missing = 1 - prob_mass

    prob_to_add = mass_missing / (VOCAB_LENGTH - topk)
    new_distribution = np.zeros(VOCAB_LENGTH)

    np.add.at(new_distribution, indices, distribution)
    zero_indices = np.where(new_distribution == 0)[0]
    np.add.at(new_distribution, zero_indices, prob_to_add)

    return new_distribution
