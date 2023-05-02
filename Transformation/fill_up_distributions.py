import numpy as np
VOCAB_LENGTH = 16384


def fill_multiple_distributions(distributions, indices, topk=256):
    current_sum = np.sum(distributions, axis=-1, keepdims=True)

    diff = current_sum - 0.9999999
    x = np.clip(diff, 0, None)  # if value > 0 do  nothing, if value <= 0, 0)

    distributions_new = distributions.copy()

    distributions_new[..., 0] -= x[..., 0]

    current_sum = np.sum(distributions_new, axis=-1, keepdims=True)

    needed_sum = 1 - current_sum

    num_probs_to_fill = VOCAB_LENGTH - topk  # this is 16128

    prop_to_fill_per_element = needed_sum / num_probs_to_fill

    new_probs = np.full((distributions_new.shape[0], distributions_new.shape[1], VOCAB_LENGTH), prop_to_fill_per_element)

    indices = indices.astype(int)

    depth = np.indices(distributions_new.shape)[0]
    rows = np.indices(distributions_new.shape)[1]

    new_probs[depth, rows, indices] = distributions_new

    return new_probs


if __name__ == "__main__":
    np.random.seed(3)
    a = np.random.random((2, 2, 3))
    a = np.divide(a, np.sum(a, axis=-1, keepdims=True))
    a[0, 0, 0] = 0.4
    a[1, 0, 1] = 0.3

    print(f"a: {a}")
    print(f"sums of a: {np.sum(a, axis=-1)}")

    indices = np.indices(a.shape)[-1]
    #indices[0, 0, 1] = 3

    print(f"indices: {indices}")

    new_probs = fill_multiple_distributions(a, indices, topk=3)

    print(f"new probs: {new_probs}")

    print(f"sum new probs: {np.sum(new_probs, axis=-1)}")
