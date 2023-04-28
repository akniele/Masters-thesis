import numpy as np
VOCAB_LENGTH = 16384


def fill_multiple_distributions(distributions, indices, topk=256):
    current_sum = np.sum(distributions, axis=-1)

    needed_sum = 1 - current_sum

    num_probs_to_fill = VOCAB_LENGTH - topk

    prop_to_fill_per_element = needed_sum / num_probs_to_fill

    prop_to_fill_per_element = np.expand_dims(prop_to_fill_per_element, -1)

    new_probs = np.full((distributions.shape[0], distributions.shape[1], VOCAB_LENGTH), prop_to_fill_per_element)

    indices = indices.astype(int)

    depth = np.arange(len(distributions))
    depth = np.expand_dims(depth, 1)
    depth = np.expand_dims(depth, 2)
    depth = np.broadcast_to(depth, distributions.shape)

    rows = np.arange(distributions.shape[1])
    rows = np.expand_dims(rows, 1)
    rows = np.broadcast_to(rows, distributions.shape)

    new_probs[depth, rows, indices] = distributions

    return new_probs
