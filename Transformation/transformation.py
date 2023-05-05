import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
import pickle
import sys
import warnings

sys.path.append('../GetClusters')
#from ClassifierFiles.classifier import BertClassifier
from scipy.stats import entropy # kl-divergence/relative entropy if optional parameter qk is given, else calculate Shannon entropy
from Transformation.fill_up_distributions import fill_multiple_distributions


# Fredrik's metric, for this purpose timeStepDiffs and sampleDiffs returns the same
def weightedManhattanDistance(dist1, dist2, probScaleLimit=0.02):
    probSums = dist1 + dist2
    belowThreshold = np.where(probSums < probScaleLimit, 1, 0)
    belowThresholdMask = (belowThreshold / probScaleLimit) * probSums
    overThresholdMask = 1 - belowThreshold
    weightMask = belowThresholdMask + overThresholdMask

    absDiff = np.abs(dist1 - dist2) * weightMask
    timeStepDiffs = np.sum(absDiff, axis=-1)
    sampleDiffs = np.sum(timeStepDiffs, axis=-1)

    return absDiff, timeStepDiffs, sampleDiffs


def f(beta, p, entropy_small, counter):  # solution found here: https://stats.stackexchange.com/questions/521582/controlling-the-entropy-of-a-distribution
    z = sum(p**beta)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            new_entropy = (-1 / z) * sum((p**beta) * (beta * np.log(p) - np.log(z)))
        except Warning as e:
            print(f"error found: {e}")
            print(f" current value of beta: {beta}")
            print(f" entropy of current distribution: {entropy(np.squeeze(p, -1))}")
            np.save(f'/home/ubuntu/pipeline/plots/broken_distr_{counter[0]}', np.squeeze(p, -1))
            counter[0] += 1
            return 1

    return (new_entropy - entropy_small)**2


def trans_1(bigprobs, mean_entropy, upper_bound):
    """
    :param bigprobs: a single probability distribution from the big model
    :param smallprobs: a single probability distribution from the small model
    :param mean_entropy: the mean entropy of all distributions from the small model (from training data)
    :return: the probability distribution from the big model, transformed to approximate the entropy of the small model
    """
    # change the entropy of the big probability distribution to make it more similar to the entropy of the smaller model

    p = bigprobs.astype('float64')
    big_entropy = entropy(bigprobs)
    p = np.expand_dims(p, 1)  # unsqueeze p (optimizer wants (n,1))

    small_entropy = big_entropy - mean_entropy

    bounds = [(0, upper_bound)]

    counter = [0]

    solution = minimize(fun=f, x0=1, bounds=bounds, args=(p, small_entropy, counter))  # find minimum of function f, initial guess is set to 1 because prob**1 is just prob
    new_z = sum(p**solution.x)
    transformed_p = (p**solution.x) / new_z
    transformed_p = np.squeeze(transformed_p, 1)  # squeeze away the dimension we needed for the optimizer
    return transformed_p  # return the big model's probs, transformed to have the same entropy as the small model's probs


def trans_0(bigprobs, mean_bucket_trans, bucket_indices):
    print(f"number of zeros in array: {(bigprobs.size - np.count_nonzero(bigprobs))}")
    print(f"number of negative values in array: {np.sum(bigprobs < 0)}")

    bigprobs = np.expand_dims(bigprobs, 0)  # array has only two dimensions at this point

    sorted_indices = np.argsort(bigprobs, axis=-1, kind='stable')[:, :, ::-1]

    depth, rows = np.indices(bigprobs.shape)[:2]

    sorted_big_probs = bigprobs[depth, rows, sorted_indices]

    print(f"number of zeros in array: {(bigprobs.size - np.count_nonzero(bigprobs))}")
    print(f"number of negative values in array: {np.sum(bigprobs < 0)}")

    #del bigprobs

    current_bucket_probs = np.ones((sorted_big_probs.shape[0], sorted_big_probs.shape[1], len(bucket_indices)-1))

    for i, index in enumerate(bucket_indices[:-1]):  # get current bucket probabilities
        current_bucket_probs[:, :, i] = np.sum(sorted_big_probs[:, :, index:bucket_indices[i + 1]], -1)

    new_bucket_trans = current_bucket_probs + mean_bucket_trans  # add up current bucket probs and transformations from train data

    min_cols = np.amin(new_bucket_trans, axis=-1)  # get min bucket prob

    output = min_cols < 0  # create boolean mask, True if there's a bucket prob < 0, else False

    def add_min_and_normalize(x, epsilon=10e-10):  # called if there's a bucket prob < 0, adds this prob to all probs, then normalizes
        x2 = x + epsilon + np.expand_dims(abs(np.min(x, axis=-1)), -1)  # this can give zero probability
        return x2 / np.expand_dims(np.sum(x2, axis=-1), -1)

    def normalize(x):  # normalizes the probabilities (squeezes them between 0 and 1)
        return x / np.expand_dims(np.sum(x, axis=-1), -1)

    new_bucket_trans[output, :] = add_min_and_normalize(new_bucket_trans[output, :])  # apply add_min_and_normalize to rows with bucket prob < 0
    new_bucket_trans[~output, :] = normalize(new_bucket_trans[~output, :])  # apply normalize function to rows without bucket prob < 0

    target_transformation = new_bucket_trans - current_bucket_probs  # get final bucket transformation, i.e. how much to add / subtract from each bucket after normalization

    for i, index in enumerate(bucket_indices[:-1]):  # get current bucket probabilities
        sorted_big_probs[:, :, index:bucket_indices[i+1]] = sorted_big_probs[:, :, index:bucket_indices[i + 1]] + \
                                                     np.expand_dims(target_transformation[:, :, i] /
                                                                    (bucket_indices[i + 1] - index), -1)

    sorted_big_probs = add_min_and_normalize(sorted_big_probs)

    print(f"after transformation number of zeros in array: {(sorted_big_probs.size - np.count_nonzero(sorted_big_probs))}")
    print(f"after transformation number of negative values in array: {np.sum(sorted_big_probs < 0)}")

    final_probs = np.zeros(sorted_big_probs.shape)
    final_probs[depth, rows, sorted_indices] = sorted_big_probs  # unsort the probabilities

    final_probs = np.squeeze(final_probs, 0)

    print(f"after sorting number of zeros in array: {(final_probs.size - np.count_nonzero(final_probs))}")
    print(f"after sorting number of negative values in array: {np.sum(final_probs < 0)}")

    return final_probs


def trans_2(probs, mean_k, top_p):
    probs = np.expand_dims(probs, 0)
    sorted_indices = np.argsort(probs, axis=-1, kind='stable')[:, :, ::-1]

    depth = np.indices(probs.shape)[0]
    rows = np.indices(probs.shape)[1]

    print("here?")
    sorted_probs = probs[depth, rows, sorted_indices]  # sort probabilities descendingly, shape: (1, num_distr, 16384)

    print(f"data type of sorted_probs: {sorted_probs.dtype}")
    print(f"shape of sorted probs, after sorting: {sorted_probs.shape}")
    print(f"sum of sorted probs, random distribution: {np.sum(sorted_probs[0,5])}")
    print(f"smallest element in sorted_probs: {np.min(sorted_probs)}")

    print("or here?")
    cumsum = np.cumsum(sorted_probs, axis=-1)  # get the cumulative sum of each distribution
    mask = cumsum >= top_p
    print("maybe somewhere here?")
    current_k = np.argmax(mask, axis=-1)  # get the number of tokens the top-p probability mass is currently occupying
    current_k = np.expand_dims(current_k, -1)

    print("what about there?")
    target_k = current_k - mean_k  # get the number of tokens we want top-p to be distributed over

    target_k[target_k < 0] = 0  # replace negative values with 0 (target number of elements k needs to be at least 0)

    print("or in this place?")
    idx_array = np.arange(sorted_probs.shape[-1])
    idx_array = np.broadcast_to(idx_array, sorted_probs.shape)

    print("maybe this is where it is?")
    mask = idx_array <= target_k  # create a boolean mask that is True if the index is bigger than the target index

    current_sum = np.sum(sorted_probs, axis=-1, where=mask)
    print("or could be here?")
    scaling_factor = 0.9 / (current_sum + np.finfo(float).eps)
    print("is it the stupid scaling factor?")
    scaling_factor = np.reshape(scaling_factor, (sorted_probs.shape[0], sorted_probs.shape[1], 1))
    #print(f"scaling factor: {scaling_factor}")
    print("or the tmp arrays?")
    tmp = np.zeros(sorted_probs.shape)
    print("maybe it's the temps")
    tmp[mask] = sorted_probs[mask]
    #print(f"tmp: {tmp}")
    tmp *= scaling_factor
    del scaling_factor
    #print(f"tmp after scaling: {tmp}")

    print("second part of the function")
    tmp_2 = np.zeros(sorted_probs.shape)
    print("going strong")
    current_sum_2 = 1 - current_sum
    print("last one")

    print(f"number of zeros in array: {(current_sum_2.size - np.count_nonzero(current_sum_2))}")
    scaling_factor_2 = 0.1 / (current_sum_2 + 10e-10)  #np.finfo(float).eps
    print("here we are")
    del current_sum_2
    scaling_factor_2 = np.reshape(scaling_factor_2, (sorted_probs.shape[0], sorted_probs.shape[1], 1))
    print("and now here")
    inverted_mask = np.invert(mask)  # here it crashes!
    del mask
    tmp_2[inverted_mask] = sorted_probs[inverted_mask]
    print("almost done now")
    tmp_2 *= scaling_factor_2
    print("finishline")
    del scaling_factor_2
    #print(f"tmp_2: after scaling: {tmp_2}")

    print("adding up the two arrays")
    final_array = tmp + tmp_2
    #print(f"final_array: {final_array}")

    # cumsum = np.cumsum(sorted_probs, axis=-1)  # get the cumulative sum of each distribution
    # mask = cumsum >= top_p
    # current_k = np.argmax(mask, axis=-1)  # get the number of tokens the top-p probability mass is currently occupying
    # current_k = np.expand_dims(current_k, -1)
    #
    # target_k = current_k - mean_k  # get the number of tokens we want top-p to be distributed over
    #
    # target_k[target_k < 0] = 0  # replace negative values with 0 (target number of elements k needs to be at least 0)
    #
    # idx_array = np.arange(sorted_probs.shape[-1])
    # idx_array = np.broadcast_to(idx_array, sorted_probs.shape)
    #
    # mask = idx_array <= target_k  # create a boolean mask that is True if the index is bigger than the target index
    #
    # sum_top_p = np.sum(sorted_probs, axis=-1, where=mask, keepdims=True)  # get the sum of the target_k probabilities
    # sum_not_top = np.sum(sorted_probs, axis=-1, where=np.invert(mask), keepdims=True)  # get the inverse
    #
    # sorted_probs = np.divide(sorted_probs, sum_top_p, where=mask, out=np.zeros_like(sorted_probs)) # divide the top-k probabilities by their current sum
    # print(f"number of zeros in array: {(sorted_probs.size - np.count_nonzero(sorted_probs))}")
    #
    # sorted_probs = np.multiply(sorted_probs, top_p, where=mask, out=np.zeros_like(sorted_probs))  # divide them by the target sum
    # print(f"number of zeros in array: {(sorted_probs.size - np.count_nonzero(sorted_probs))}")
    #
    # sorted_probs = np.divide(sorted_probs, sum_not_top, where=np.invert(mask), out=np.zeros_like(sorted_probs))  # same for the other probabilities
    # print(f"number of zeros in array: {(sorted_probs.size - np.count_nonzero(sorted_probs))}")
    #
    # sorted_probs = np.multiply(sorted_probs, 1 - top_p, where=np.invert(mask), out=np.zeros_like(sorted_probs))
    # print(f"number of zeros in array: {(sorted_probs.size - np.count_nonzero(sorted_probs))}")

    print("sorting...")
    final_probs = np.zeros(final_array.shape)
    final_probs[depth, rows, sorted_indices] = final_array  # unsort the probabilities
    print(f"number of zeros in array: {(final_probs.size - np.count_nonzero(final_probs))}")

    final_probs = np.squeeze(final_probs, 0)

    return final_probs


def transformations(bigprobs, indices, mean_features, num_features, bucket_indices, function,
                    upper_bound, top_p, pred_labels=None):
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

    print(f"mean features: {mean_features}")

    print(f"zeros before filling up distribution: {np.min(bigprobs)}")
    print("  => FILL UP DISTRIBUTIONS")
    filled_up_probs = fill_multiple_distributions(bigprobs, indices)
    print("  => DONE FILLING UP DISTRIBUTIONS")
    print(f"zeros after filling up distribution: {np.min(filled_up_probs)}")

    transformed_probs = filled_up_probs.copy()

    if pred_labels is not None:
        unique_labels = np.unique(pred_labels)

        for label in tqdm(unique_labels, desc="iterating over labels"):
            means = []
            for j in range(num_features):
                means.extend([mean_features[f"{function.__name__}_{j}_{label}"]])

            print(f"means for label {label}: {means}")

            if function.__name__ == "get_entropy_feature":
                print("  => ENTROPY TRANSFORMATION")
                transformed_probs[pred_labels == label] = np.apply_along_axis(trans_1, -1,
                                                                              transformed_probs[pred_labels == label],
                                                                              means[:1], upper_bound)
            elif function.__name__ == "bucket_diff_top_k":
                print("  => BUCKET PROBABILITY TRANSFORMATION")

                transformed_probs[pred_labels == label] = trans_0(transformed_probs[pred_labels == label],
                                                                  means[:num_features], bucket_indices)

            elif function.__name__ == "get_top_p_difference":
                print("  => TOP-P PROBABILITY TRANSFORMATION")
                transformed_probs[pred_labels == label] = trans_2(transformed_probs[pred_labels == label],
                                                                  means[:1], top_p=top_p)

            else:
                raise Exception(f"{function.__name__} is not a valid transformation function.")

        print(f"  => FINISHED TRANSFORMATIONS!")

    else:
        means = []
        for j in range(num_features):
            means.extend([mean_features[f"{function.__name__}_{j}"]])

        if function.__name__ == "get_entropy_feature":
            print("  => ENTROPY TRANSFORMATION")
            transformed_probs = np.apply_along_axis(trans_1, -1, transformed_probs, means[:1], upper_bound)

        elif function.__name__ == "bucket_diff_top_k":
            print("  => BUCKET PROBABILITY TRANSFORMATION")
            transformed_probs = trans_0(transformed_probs, means[:num_features], bucket_indices)

        elif function.__name__ == "get_top_p_difference":
            print("  => TOP-P PROBABILITY TRANSFORMATION")
            transformed_probs = trans_2(transformed_probs, means[:1], top_p=top_p)

        else:
            raise Exception(f"{function.__name__} is not a valid transformation function.")

        print(f"  => FINISHED TRANSFORMATIONS!")

    return transformed_probs, filled_up_probs


def get_distances(transformed_probs, bigprobs, smallprobs):
    """
    1. Compare the transformed probs to the small probs (using the average of the weighted Manhattan distance)
    2. Compare the original big probs to the small probs (using the average of the weighted Manhattan distance)
    3. See if distance between transformed and small probs is smaller than between original big probs and small probs
    4. return mean distance between trans and small probs minus mean distance between big and small probs
    :param small_indices:
    :param transformed_probs:
    :param bigprobs:
    :param smallprobs:
    :return:
    """

    _, dist_trans_tmp, _ = weightedManhattanDistance(transformed_probs, smallprobs, probScaleLimit=0.02)
    _, dist_big_tmp, _ = weightedManhattanDistance(bigprobs, smallprobs, probScaleLimit=0.02)

    return dist_trans_tmp, dist_big_tmp


def get_mean_distances(dist_trans, dist_big, filename):
    mean_dist_trans = np.mean(dist_trans).item()
    mean_dist_big = np.mean(dist_big).item()

    std_dist_trans = np.std(dist_trans).item()
    std_dist_big = np.std(dist_big).item()

    with open(f"/home/ubuntu/pipeline/logfiles/{filename}.txt", "a") as logfile:
        logfile.write(f"Mean Weighted Manhattan Distance between transformed distributions and target: "
                      f"{mean_dist_trans}\n"
                      f"Mean Weighted Manhattan Distance between untransformed distributions and target: "
                      f"{mean_dist_big}\n"
                      f"Standard Deviation of the Weighted Manhattan Distances of the transformed distributions: "
                      f"{std_dist_trans}\n"
                      f"Standard Deviation of the Weighted Manhattan Distances of the untransformed distributions: "
                      f"{std_dist_big}\n")

    return mean_dist_trans - mean_dist_big, std_dist_trans - std_dist_big


if __name__ == "__main__":
    probs0 = np.array([[[0.2, 0.004, 0.01]]])
    probs1 = np.array([[[0.5, 0.001, 0.005]]])
    print(f"probs0: {probs0}")
    print(f"probs1: {probs1}")

    _, diff, _ = weightedManhattanDistance(probs1, probs0)
    print(diff)
