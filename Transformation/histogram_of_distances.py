# histogram of the weighted Manhattan Distances, before and after transformation
import numpy as np
from matplotlib import pyplot as plt


def difference_histogram(trans_dist, original_dist, filename):
    trans = np.squeeze(trans_dist, -1)
    orig = np.squeeze(original_dist, -1)

    trans = trans.flatten()
    orig = orig.flatten()
    print(trans.shape)

    plt.hist(trans, bins=20, alpha=0.5, label="transformed")
    plt.hist(orig, bins=20, alpha=0.5, label="original")
    plt.legend(loc="upper right")
    plt.ylabel('Frequency')
    plt.xlabel('Differences in weighted Manhattan distance')
    plt.title("Comparing the distribution of differences")
    plt.savefig(f"plots/difference_plot_{filename}.png")
    plt.close()
