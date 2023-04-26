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


def f(beta, p, entropy_small):  # solution found here: https://stats.stackexchange.com/questions/521582/controlling-the-entropy-of-a-distribution
    z = sum(p**beta)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            new_entropy = (-1 / z) * sum((p**beta) * (beta * np.log(p) - np.log(z)))
        except Warning as e:
            print(f"error found: {e}")
            print(f" current value of beta: {beta}")
            print(f" entropy of current distribution: {entropy(np.squeeze(p, -1))}")
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

    solution = minimize(fun=f, x0=1, bounds=bounds, args=(p, small_entropy))  # find minimum of function f, initial guess is set to 1 because prob**1 is just prob
    new_z = sum(p**solution.x)
    transformed_p = (p**solution.x) / new_z
    transformed_p = np.squeeze(transformed_p, 1)  # squeeze away the dimension we needed for the optimizer
    return transformed_p  # return the big model's probs, transformed to have the same entropy as the small model's probs


def trans_0(bigprobs, mean_bucket_trans, bucket_indices):

    bigprobs = np.expand_dims(bigprobs, 0)

    sorted_indices = np.argsort(bigprobs, axis=-1, kind='stable')[:, :, ::-1]

    depth = np.arange(len(bigprobs))
    depth = np.expand_dims(depth, 1)
    depth = np.expand_dims(depth, 2)
    depth = np.broadcast_to(depth, bigprobs.shape)

    rows = np.arange(bigprobs.shape[1])
    rows = np.expand_dims(rows, 1)
    rows = np.broadcast_to(rows, bigprobs.shape)

    sorted_big_probs = bigprobs[depth, rows, sorted_indices]

    del bigprobs

    current_bucket_probs = np.ones((sorted_big_probs.shape[0], sorted_big_probs.shape[1], len(bucket_indices)-1))

    for i, index in enumerate(bucket_indices[:-1]):  # get current bucket probabilities
        current_bucket_probs[:, :, i] = np.sum(sorted_big_probs[:, :, index:bucket_indices[i + 1]], -1)

    new_bucket_trans = current_bucket_probs + mean_bucket_trans  # add up current bucket probs and transformations from train data

    min_cols = np.amin(new_bucket_trans, axis=-1)  # get min bucket prob

    output = min_cols < 0  # create boolean mask, True if there's a bucket prob < 0, else False

    def add_min_and_normalize(x):  # called if there's a bucket prob < 0, adds this prob to all probs, then normalizes
        x2 = x + np.expand_dims(abs(np.min(x, axis=-1)), -1)
        return x2 / np.expand_dims(np.sum(x2, axis=-1), -1)

    def normalize(x):  # normalizes the probabilities (squeezes them between 0 and 1)
        return x / np.expand_dims(np.sum(x, axis=-1), -1)

    new_bucket_trans[output, :] = add_min_and_normalize(new_bucket_trans[output, :])  # apply add_min_and_normalize to rows with bucket prob < 0
    new_bucket_trans[~output, :] = normalize(new_bucket_trans[~output, :])  # apply normalize function to rows without bucket prob < 0

    target_transformation = new_bucket_trans - current_bucket_probs # get final bucket transformation, i.e. how much to add / subtract from each bucket after normalization

    for i, index in enumerate(bucket_indices[:-1]):  # get current bucket probabilities
        sorted_big_probs[:, :, index:bucket_indices[i+1]] = sorted_big_probs[:, :, index:bucket_indices[i + 1]] + \
                                                     np.expand_dims(target_transformation[:, :, i] /
                                                                    (bucket_indices[i + 1] - index), -1)

    final_probs = np.zeros(sorted_big_probs.shape)  # TODO: Check if this, and the unsorting, does what it is supposed to!
    final_probs[depth, rows, sorted_indices] = sorted_big_probs  # unsort the probabilities

    final_probs = np.squeeze(final_probs, 0)

    return final_probs


def trans_2(probs, mean_k, top_p):  # transform probabilities
    cumsum = np.cumsum(probs, axis=-1)
    mask = cumsum >= top_p
    if np.any(mask):
        current_k = np.argmax(mask) + 1
        target_k = current_k - mean_k

        indices = np.array([0, target_k, probs.shape[-1]])

        #current_top_p_sum = np.sum(probs[indices[0]: indices[1]], axis=-1)

        target_p = np.array([top_p, 1 - top_p])

        for i, index in enumerate(indices[:-1]):
            probs[indices[i]: indices[i + 1]] = probs[indices[i]: indices[i + 1]] / np.sum(
                probs[indices[i]: indices[i + 1]],
                axis=-1) * target_p[i]

        return probs

    else:
        print("Raise some sort of Error!")
        return False


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
    print(f"type mean features: {type(mean_features)}")

    print("  => FILL UP DISTRIBUTIONS")
    filled_up_probs = fill_multiple_distributions(bigprobs, indices)
    print("  => DONE FILLING UP DISTRIBUTIONS")

    transformed_probs = filled_up_probs.copy()
    print(f"transformed probs shape: {transformed_probs.shape}")

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


def get_distances(transformed_probs, bigprobs, smallprobs, small_indices):
    """
    1. Fill up distributions from the smaller model
    2. Compare the transformed probs to the small probs (using the average of the weighted Manhattan distance)
    3. Compare the original big probs to the small probs (using the average of the weighted Manhattan distance)
    4. See if distance between transformed and small probs is smaller than between original big probs and small probs
    5. return mean distance between trans and small probs minus mean distance between big and small probs
    :param transformed_probs:
    :param bigprobs:
    :param smallprobs:
    :return:
    """

    filled_up_small_probs = fill_multiple_distributions(smallprobs, small_indices)

    _, dist_trans_tmp, _ = weightedManhattanDistance(transformed_probs, filled_up_small_probs, probScaleLimit=0.02)
    _, dist_big_tmp, _ = weightedManhattanDistance(bigprobs, filled_up_small_probs, probScaleLimit=0.02)

    print(f"shape dist_trans tmp: {dist_trans_tmp.shape}")
    print(f"shape dist_big tmp: {dist_big_tmp.shape}")

    print(f"type dist_big tmp: {type(dist_big_tmp)}")
    print(f"dtype dist_trans tmp: {dist_trans_tmp.dtype}")

    return dist_trans_tmp, dist_big_tmp


def get_mean_distances(dist_trans, dist_big):
    mean_dist_trans = np.mean(dist_trans).item()
    mean_dist_big = np.mean(dist_big).item()

    print(f" mean_dist_trans: {mean_dist_trans}")
    print(f" mean_dist_big: {mean_dist_big}")

    return mean_dist_trans - mean_dist_big


if __name__ == "__main__":
    # f = lambda x: torch.tensor(x, dtype=torch.float32)
    #
    # unchanged = f([0.3, 0.2, 0.5])
    # print("Unchanged", unchanged)
    #
    # transformation = f([-.2, 0.3, -0.3])
    # print("Transformation", transformation)
    #
    # actual_trans = trans_0(unchanged, transformation, indices=[1, 4])

    # arr = np.array([[[0.3, 0.3, 0.4],
    #                  [0.3, 0.9, 0.4],
    #                  [0.1, 0.7, 0.4]],
    #
    #                 [[0.3, 0.2, 0.9],
    #                  [0.1, 0.1, 0.8],
    #                  [0.3, 0.2, 0.5]]])
    #
    # mean_bucket_trans = np.array([-0.2, 0.4])
    # mean_bucket_trans = np.expand_dims(mean_bucket_trans, 0)
    # mean_bucket_trans = np.expand_dims(mean_bucket_trans, 0)
    #
    # indices = [0, 1, 3]
    #
    # target_probs = trans_0(arr, mean_bucket_trans, indices)
    #
    # print(target_probs)


    # NUM_TRAIN_SHEETS = 20
    # NUM_TEST_SHEETS = 12
    #
    """Load data"""
    # data
    probs1 = pickle.load(open("../train_data/train_big_100000_0.pkl", "rb"))
    indices = pickle.load(open("../train_data/indices_big_100000_0.pkl", "rb"))
    probs1 = probs1.detach().numpy()
    indices = indices.detach().numpy()

    print(f"shape probs: {probs1.shape}")

    probs1 = fill_multiple_distributions(probs1[:500], indices[:500], topk=256)

    mean_bucket_trans = np.array([[[-0.2, 0.5, 0.4]]])

    bucket_indices = [0, 20, 50, probs1.shape[-1]]

    trans_probs = trans_0(probs1, mean_bucket_trans, bucket_indices)

    print(trans_probs.shape)
    print(trans_probs[:1, :, :])

    # print(f"entropy small model: {entropy(probs0[0][0])}")
    # print(f"entropy big model: {entropy(probs1[0][0])}")
    # trans_p = trans1(probs1[0][0], probs0[0][0], 0.0210886001586914)
    # print(entropy(trans_p))



    #
    # """transform big probs to be more similar to small probs"""
    #
    # transformed_probs = probability_transformation(probs1[:20, :, :], probs0[:20, :, :])
    # print(transformed_probs.shape)
    #
    # total_weighted_changed, total_weighted_unchanged = compare_distance(transformed_probs, probs0[:20, :, :], probs1[:20, :, :])
    #
    # print(f"standard deviation of weighted Manhattan distances after transformation: {np.std(total_weighted_changed)}")
    # print(f"standard deviation of weighted Manhattan distances without transformation: {np.std(total_weighted_unchanged)}")
    #
    # print(f"mean of weighted Manhattan distances after transformation: {np.mean(total_weighted_changed)}")
    # print(
    #     f"mean of weighted Manhattan distances without transformation: {np.mean(total_weighted_unchanged)}")
    #
    #
    # # print(f"sum of Manhattan Distance after transformation: {total_changed}")
    # # print(f"sum of Manhattan Distance without transformation: {total_unchanged}")
    #
    # print(f"sum of weighted Manhattan Distance after transformation: {sum(total_weighted_changed)}")
    # print(f"sum of weighted Manhattan Distance without: {sum(total_weighted_unchanged)}")
    #
    # #transformed_probs = probability_transformation(probs1, probs0)
    #
    # improvement_weighted = compare_distributions(transformed_probs, probs0[:20, :, :], probs1[:20, :, :])
    #
    # print(f" Using the Manhattan distance, % "  #{improvement_manhattan * 100}
    #       f"of the transformed distributions are closer to the target distribution.\nFor the weighted Manhattan "
    #       f"distance, the score is {improvement_weighted * 100}%.")
    #
    # improvement_manhattan_128, improvement_weighted_128 = sanity_check_distance_metrics(transformed_probs,
    #                                                                                     probs0[:20, :, :], probs1[:20, :, :])
    # print(f" Using the cutoff Manhattan distance, {improvement_manhattan_128 * 100}% "
    #       f"of the transformed distributions are closer to the target distribution.\nFor the cutoff weighted Manhattan "
    #       f"distance, the score is {improvement_weighted_128 * 100}%.")
