"""
# Plan

- Define Distribution Difference Metrics
- Calculate/Cluster Distributions into families
- Use the calculated distributions as training labels for classifier
- Train a classifier, that takes a probability distribution and outputs the target family (label from above)
- Calculate the family-transformation for each family.
- Evaluate the given family-classifier and transformations on held-out data
"""
import sys
import pickle
import statistics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
from tqdm import tqdm
import math
import torch
import scipy.stats as stats
import matplotlib.pyplot as plt

from GetClusters.clustering import k_means_clustering, label_distribution
from GetClusters.clustering import find_optimal_n
from GetClusters.differenceMetrics import bucket_diff_top_k
from GetClusters.differenceMetrics import get_entropy_feature
from GetClusters.featureVector import load_feature_vector
from Transformation.get_means_from_training_data import get_means_from_training_data
from Transformation.fill_up_distributions import fill_distribution
from GetClusters.featureVector import create_and_save_feature_vector
from ClassifierFiles import train_and_evaluate_classifier
from GetClusters.featureVector import fill_all_distributions_and_create_features
from ClassifierFiles.train_and_evaluate_classifier import make_predictions
from Transformation.transformation import transformations, get_distances, get_mean_distances


sys.path.insert(1, '../Non-Residual-GANN')
from GenerateData.data_sorted_by_big_model import generateData
# from Transformation.get_means_from_training_data import get_mean_entropy_from_training_data
# from Transformation.transformation import classifyProbabilityIntoFamily, create_prediction_data, transformProbabilities
# from Transformation.transformation import trans0
# from Transformation.transformation import probability_transformation
# from Transformation.transformation import compare_distributions


def train(functions, bucket_indices, num_clusters, batch_size, epochs, lr, num_test_samples,
             generate_data=False, train_classifier=False):

    NUM_TRAIN_SHEETS = 10_000  # for creating feature vectors, and training classifier
    N_CLUSTERS = num_clusters  # number of clusters used for clustering

    # for classifier:
    NUM_CLASSES = N_CLUSTERS  # number of classes for the classifier model (has to be same as number of clusters)
    BATCH_SIZE = batch_size
    EPOCHS = epochs
    LR = lr

    if generate_data:
        print("  => GENERATING DATA AND FEATURE VECTORS")
        generateData(functions, bucket_indices, num_samples=100_000, truncate=True, topk=256, save=True)

    print("  => LOADING SCALED FEATURES FOR CLUSTERING")
    scaled_features = load_feature_vector(functions=functions, num_features=4, num_sheets=NUM_TRAIN_SHEETS, scaled=True)
    print(f"Shape of scaled features: {scaled_features.shape}")
    print(scaled_features[:3])
    print(scaled_features[(NUM_TRAIN_SHEETS*64)-1:(NUM_TRAIN_SHEETS*64)+3])

    print("  => CLUSTERING")
    labels = k_means_clustering(scaled_features, N_CLUSTERS)

    labels_distribution = label_distribution(N_CLUSTERS, labels)
    print(f"label distribution: {labels_distribution}")

    # Chi-Square Goodness of Fit Test
    chi_square_test_statistic, p_value = stats.chisquare(
        labels_distribution, [(sum(labels_distribution)/len(labels_distribution)) for i in range(len(labels_distribution))])

    # chi square test statistic and p value
    print(f"chi_square_test_statistic is : {chi_square_test_statistic}")
    print(f"p_value : {p_value:.3e}")

    # find Chi-Square critical value
    print(stats.chi2.ppf(1 - 0.05, df=2))

    print("  => GET MEAN FEATURES FROM TRAINING DATA")
    dict_means = get_means_from_training_data(functions=functions, num_features=4, num_sheets=NUM_TRAIN_SHEETS,
                                              labels=labels)

    for key, value in dict_means.items():
        print(f"key: {key}\t value:{value}")

    if train_classifier:
        pred_labels, true_labels = train_and_evaluate_classifier.train_and_evaluate_classifier(
            NUM_CLASSES, BATCH_SIZE, EPOCHS, LR, labels, NUM_TRAIN_SHEETS)

    new_pred_labels = make_predictions(3, num_sheets=num_test_samples)

    print(f"pred labels: {new_pred_labels[:50]}")

    new_pred_labels = np.reshape(new_pred_labels[:num_test_samples*64], (num_test_samples, 64))

    print(f"new shape of pred labels: {new_pred_labels.shape}")

    return new_pred_labels, dict_means


def test(new_pred_labels, dict_means, num_test_samples, bucket_indices, functions):
    print("  => LOAD DATA FOR TRANSFORMATION")

    with open(f"train_data/train_big_100000_9.pkl", "rb") as f:
        bigprobs = pickle.load(f)

    bigprobs = bigprobs[:num_test_samples].numpy()

    #max_probs = np.amax(bigprobs, axis=-1)
    #min_of_max_probs = np.amin(max_probs)

    #print(f"min of max probs: {min_of_max_probs}")

    #upper_bound = - (math.log(1.7976931348623157e+308, min_of_max_probs))

    with open(f"train_data/indices_big_100000_9.pkl", "rb") as g:
        indices1 = pickle.load(g)

    indices1 = indices1[:num_test_samples].numpy()

    print("  => LOAD DATA FOR EVALUATION")

    with open(f"train_data/train_small_100000_9.pkl", "rb") as f:
        smallprobs = pickle.load(f)

    smallprobs = smallprobs[:num_test_samples].numpy()

    with open(f"train_data/indices_big_100000_9.pkl", "rb") as g:
        indices0 = pickle.load(g)

    indices0 = indices0[:num_test_samples].numpy()

    bucket_indices.insert(0, 0)
    bucket_indices.append(16_384)

    return bigprobs, smallprobs, indices1, indices0, new_pred_labels, num_test_samples

    #for i in tqdm(range(num_test_samples//100)):  # so if num_test_samples == 500, then the loop has 5 iterations


def generate_data(functions, bucket_indices, num_samples=100_000, truncate=True, topk=256, save=True):
    print("  => GENERATING DATA AND FEATURE VECTORS")
    generateData(functions, bucket_indices, num_samples=num_samples, truncate=truncate, topk=topk, save=save)

import numpy as np

if __name__ == "__main__":

    with open(f"train_data/train_small_100000_0.pkl", "rb") as f:
        bigprobs = pickle.load(f)

    print('Datatype bigprobs:', bigprobs.dtype)
    print(f"bigprobs:\n {bigprobs[0, 0, :]}")

    with open(f"final_data/small_10000_0.pkl", "rb") as g:
        bigprobs_new = pickle.load(g)
        print('Datatype bigprobs_new:', bigprobs_new.dtype)
    print(f"bigprobs_new: \n  {bigprobs_new[0, 0, :]}")

    with open(f"final_data/indices_10000_0.pkl", "rb") as h:
        sorted_indices = pickle.load(h)


    # generate data
    # BUCKET_INDICES = [10, 35]
    # FUNCTIONS = [bucket_diff_top_k, get_entropy_feature]
    # print("  => GENERATING DATA AND FEATURE VECTORS")
    # generateData(functions=FUNCTIONS, bucket_indices=BUCKET_INDICES,
    #              num_samples=100_000, truncate=True, topk=256, save=True)

    # functions = [bucket_diff_top_k, get_entropy_feature]
    # NUM_TRAIN_SHEETS = 10_000
    #
    # data = load_feature_vector(functions=functions, num_features=4, num_sheets=NUM_TRAIN_SHEETS, scaled=False)
    # print(f" feature vector shape: {data.shape}")
    #
    # first_column = data[:, 0]
    # second_column = data[:, 1]
    # third_column = data[:, 2]
    #
    # fig = plt.figure(figsize=(10, 7))
    #
    # # Creating plot
    # plt.boxplot(data[:, :-1])
    #
    # # show plot
    # plt.savefig("example_plot.png")

    """Load data"""
    #NUM_TRAIN_SHEETS = 10000  # for creating feature vectors, and training classifier

    # small_probs, big_probs, small_indices_final, big_indices_final = generateData(
    #          num_samples=100000, truncate=True, topk=256, save=True)

    #bucket_indices = [10, 35]
    #functions = [bucket_diff_top_k, get_entropy_feature]

    #small_probabilities, big_probabilities, small_indices, big_indices, features = \
    #generateData(functions, bucket_indices, num_samples=100000, truncate=True, topk=256, save=True)



#########################################################################
    # for i in tqdm(range(1)):
    #     print("  => LOADING DATA")
    #     print(i)

    # with open(f"train_data/train_big_100000_9.pkl", "rb") as f:
    #     probs1 = pickle.load(f)
    #
    # probs1 = probs1.numpy()
    #
    #     with open(f"train_data/train_small_100000_{i}.pkl", "rb") as g:
    #         probs0 = pickle.load(g)
    #
    #     probs0 = probs0.numpy()
    #
    #     with open(f"train_data/indices_small_100000_{i}.pkl", "rb") as g:
    #         indices0 = pickle.load(g)
    #
    #     indices0 = indices0.numpy()
    #
    # with open(f"train_data/indices_big_100000_9.pkl", "rb") as g:
    #     indices1 = pickle.load(g)
    #
    # indices1 = indices1.numpy()
    #
    #     functions = []
    #
    #     bucket_diffs = fill_all_distributions_and_create_features(
    #         probs0[:500, :, :], probs1[:500, :, :], indices0[:500, :, :], indices1[:500, :, :], functions,
    #         num_features=1, topk=256)
    #
    #     print(bucket_diffs.shape)
    #     print(bucket_diffs[:, :, :1].shape)
    #     print(np.mean(bucket_diffs[:, :, 2:3]))

###################################################################################################333
        #print(np.sum(probs1[0][0]) - np.sum(probs0[0][0]))

    #
    #     print("  => LOADING FEATURE VECTORS")
    #
    #     """create scaled feature vector"""

    #functions = [bucket_diff_top_k, get_entropy_feature]  # difference metrics to use for creating feature vector
    # #
    # create_and_save_feature_vector(function_list, NUM_TRAIN_SHEETS, filled=True)

    # --------------------------------------------------------------------------------------#
    # features = load_feature_vector(functions=functions, num_features=4, num_sheets=NUM_TRAIN_SHEETS, scaled=False)
    #
    # print(f"Shape of features: {features.shape}")
    # print(features[:3])
    # print(features[(NUM_TRAIN_SHEETS*64)-1:(NUM_TRAIN_SHEETS*64)+3])

    # scaled_features = load_feature_vector(functions=functions, num_features=4, num_sheets=NUM_TRAIN_SHEETS, scaled=True)
    #
    # print(f"Shape of scaled features: {scaled_features.shape}")
    # print(scaled_features[:3])
    # print(scaled_features[(NUM_TRAIN_SHEETS*64)-1:(NUM_TRAIN_SHEETS*64)+3])

    # ---------------------------------------------------------------------------------------#


    # token_dict = pickle.load(open("reverseVocab.p", "rb"))
    # tokens = [token_dict[i] for i in range(len(token_dict))]

    # elbow = find_optimal_n(scaled_features)

    # ---------------------------------------------------------------------#

    # N_CLUSTERS = 3
    #
    # labels = k_means_clustering(scaled_features, N_CLUSTERS)
    #
    # label_distribution = label_distribution(N_CLUSTERS, labels)
    #
    # print(f"label distribution: {label_distribution}")
    #
    # dict_means = get_means_from_training_data(functions=functions, num_features=4, num_sheets=NUM_TRAIN_SHEETS,
    #                                           labels=labels)
    #
    # for key, value in dict_means.items():
    #     print(f"key: {key}\t value:{value}")

    # -------------------------------------------------------------------------------#

    # NUM_CLASSES = 3
    # BATCH_SIZE = 16
    # EPOCHS = 50
    # LR = 5e-5
    #
    # pred_labels, true_labels = train_and_evaluate_classifier.train_and_evaluate_classifier(
    #     NUM_CLASSES, BATCH_SIZE, EPOCHS, LR, labels, NUM_TRAIN_SHEETS)

    # ------------------------------------------------------------------------------------#

    # pred_labels = make_predictions(3, num_sheets=NUM_TRAIN_SHEETS)
    #
    # print(f"pred labels: {pred_labels[:50]}")
    #
    # transformed_probs = transformations(probs1[:5], indices1[:5], dict_means, bucket_indices, functions, pred_labels=pred_labels[:5])
    #
    # print(f"shape transformed_probs: {transformed_probs.shape}")
    # print(f" example of transformed probs: {transformed_probs[0][0][:30]}")

    # held_out_data = create_prediction_data(probs1)

    """transform big probs to be more similar to small probs"""

    #new_probs = trans0(probs1[0][2])

    """ The probability transformation function should have an argument stating which transformations should be
    carried out, and in which order. It should then, for each transformation, get the mean from the training data
    
    
    """

    # probs1 = pickle.load(open("Data/probs_1.p", "rb"))
    # probs0 = pickle.load(open("Data/probs_0.p", "rb"))
    #
    # transformed_probs = probability_transformation(probs1, probs0)
    #
    # improvement_manhattan, improvement_weighted = compare_distributions(transformed_probs, probs0, probs1)
    #
    # print(f" Using the Manhattan distance, {improvement_manhattan * 100}% "
    #       f"of the transformed distributions are closer to the target distribution.\nFor the weighted Manhattan "
    #       f"distance, the score is {improvement_weighted * 100}%.")
