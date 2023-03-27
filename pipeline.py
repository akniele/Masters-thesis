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
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
from tqdm import tqdm


from GetClusters.clustering import k_means_clustering
from GetClusters.clustering import find_optimal_n, label_distribution
from GetClusters.differenceMetrics import get_entropy_feature
from GetClusters.differenceMetrics import bucket_diff_top_k
from GetClusters.differenceMetrics import get_entropy_feature
from GetClusters.featureVector import load_feature_vector
from Transformation.get_means_from_training_data import get_means_from_training_data
from Transformation.fill_up_distributions import fill_distribution
from GetClusters.featureVector import create_and_save_feature_vector
from ClassifierFiles import train_and_evaluate_classifier
from GetClusters.featureVector import fill_all_distributions_and_create_features
from ClassifierFiles.train_and_evaluate_classifier import make_predictions


sys.path.insert(1, '../Non-Residual-GANN')
from GenerateData.load_data_new import generateData
# from Transformation.get_means_from_training_data import get_mean_entropy_from_training_data
# from Transformation.transformation import classifyProbabilityIntoFamily, create_prediction_data, transformProbabilities
# from Transformation.transformation import trans0
# from Transformation.transformation import probability_transformation
# from Transformation.transformation import compare_distributions


if __name__ == "__main__":

    """Load data"""
    NUM_TRAIN_SHEETS = 10000  # for creating feature vectors, and training classifier

    # small_probs, big_probs, small_indices_final, big_indices_final = generateData(
    #          num_samples=100000, truncate=True, topk=256, save=True)

    # bucket_indices = [10, 256]  # TODO: no 0 and len(distribution), add those two in trans0 transformation function!
    # functions = [bucket_diff_top_k, get_entropy_feature]
    #
    # small_probabilities, big_probabilities, small_indices, big_indices, features = \
    #     generateData(functions, bucket_indices, num_samples=20000, truncate=True, topk=256, save=True)

#########################################################################
    # for i in tqdm(range(1)):
    #     print("  => LOADING DATA")
    #     print(i)
    #
    #     with open(f"train_data/train_big_100000_{i}.pkl", "rb") as f:
    #         probs1 = pickle.load(f)
    #
    #     probs1 = probs1.numpy()
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
    #     with open(f"train_data/indices_big_100000_{i}.pkl", "rb") as g:
    #         indices1 = pickle.load(g)
    #
    #     indices1 = indices1.numpy()
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


    function_list = [bucket_diff_top_k, get_entropy_feature]  # difference metrics to use for creating feature vector
    # #
    # create_and_save_feature_vector(function_list, NUM_TRAIN_SHEETS, filled=True)

    # --------------------------------------------------------------------------------------#
    # features = load_feature_vector(functions=function_list, num_features=4, num_sheets=NUM_TRAIN_SHEETS, scaled=False)
    #
    # print(f"Shape of features: {features.shape}")
    # print(features[:3])
    # print(features[(NUM_TRAIN_SHEETS*64)-1:(NUM_TRAIN_SHEETS*64)+3])

    # scaled_features = load_feature_vector(functions=function_list, num_features=4, num_sheets=NUM_TRAIN_SHEETS, scaled=True)
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
    # dict_means = get_means_from_training_data(functions=function_list, num_features=4, num_sheets=NUM_TRAIN_SHEETS,
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

    pred_labels = make_predictions(3, num_sheets=NUM_TRAIN_SHEETS)

    print(pred_labels[:50])

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
