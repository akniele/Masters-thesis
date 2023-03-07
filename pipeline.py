"""
# Plan

- Define Distribution Difference Metrics
- Calculate/Cluster Distributions into families
- Use the calculated distributions as training labels for classifier
- Train a classifier, that takes a probability distribution and outputs the target family (label from above)
- Calculate the family-transformation for each family.
- Evaluate the given family-classifier and transformations on held-out data
"""

import pickle
import numpy as np
import torch
from scipy.stats import entropy
import sys

from differenceMetrics import bucket_diff_top_p, bucket_diff_top_k, compare_topk_success, compare_topk_prob
from differenceMetrics import compare_average_topk_len, entropy_difference
from featureVector import create_bucket_feature_vector, create_feature_vector, scale_feature_vector
from clustering import k_means_clustering
from clustering import find_optimal_n, label_distribution
from clustering import agglomerative_clustering

sys.path.insert(1, '../Non-Residual-GANN')
from load_data_new import generateData
from get_means_from_training_data import get_mean_entropy_from_training_data
from classifier import BertClassifier
from trainClassifier import train
from evaluateClassifier import evaluate
from trainingDataClassifier import prepare_training_data
from transformation import classifyProbabilityIntoFamily, create_prediction_data, transformProbabilities
from transformation import trans0
from transformation import probability_transformation
from transformation import compare_distributions


if __name__ == "__main__":

    """Load data"""
    NUM_TRAIN_SHEETS = 9000  # for creating feature vectors, and training classifier

    # small_probs, big_probs, small_indices_final, big_indices_final = generateData(
    #     num_samples=2000, truncate=True, topk=256, save=False)
    small_probs, small_indices_final = generateData(
             num_samples=2000, truncate=True, topk=256, save=False)

    print(len(small_probs))
    print(len(small_indices_final))
    print(small_probs[0][0])
    print(len(small_probs[0][0]))

    # token_dict = pickle.load(open("reverseVocab.p", "rb"))
    # tokens = [token_dict[i] for i in range(len(token_dict))]
    #

    # """create scaled feature vector"""
    # function_list = [bucket_diff_top_k, entropy_difference]  # difference metrics to use for creating feature vector
    #
    # feature_vector = create_feature_vector(function_list, probs0, probs1, NUM_TRAIN_SHEETS)
    #
    # scaled_features = scale_feature_vector(feature_vector)  # scale feature vector
    #
    # """find optimal number of clusters for clustering, and cluster"""
    # elbow, silhouette, calinski = find_optimal_n(scaled_features)
    #
    # N_CLUSTERS = elbow
    #
    # labels = agglomerative_clustering(scaled_features, N_CLUSTERS)
    #
    # label_distribution = label_distribution(N_CLUSTERS, labels)
    #
    # print(f"label distribution: {label_distribution}")

    # entropies = get_mean_entropy_from_training_data(NUM_TRAIN_SHEETS, probs0, labels, N_CLUSTERS)
    #
    # for i in range(len(entropies.keys())):
    #     print(f"mean entropy cluster {i}: {np.mean(entropies[f'entropy_{i}'])}")
    #
    # data = create_feature_vector([bucket_diff_top_k], probs0, probs1, NUM_TRAIN_SHEETS)
    # mean_1 = -(np.mean(data[:, 0]))  # added minus because the bucket_diff_top_k returns how much more the big bucket
    #                                  # has than the small model
    # mean_2 = -(np.mean(data[:, 1]))
    # mean_3 = -(np.mean(data[:, 2]))
    #
    # print(mean_1)
    # print(mean_2)
    # print(mean_3)
    #
    #
    #
    # """train classifier on big probs and labels from clustering step"""
    # NUM_CLASSES = N_CLUSTERS
    # BATCH_SIZE = 16
    # EPOCHS = 2
    # LR = 5e-5
    #
    # df = prepare_training_data(probs1, labels, NUM_TRAIN_SHEETS)
    #
    #
    # np.random.seed(112)
    # torch.manual_seed(0)
    # df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
    #                                      [int(.8 * len(df)), int(.95 * len(df))])
    #
    # model = BertClassifier(NUM_CLASSES)
    #
    # train(model, df_train, df_val, LR, EPOCHS, BATCH_SIZE)
    #
    # torch.save(model.state_dict(), f'first_try.pt')
    #
    # """test classifier"""
    # evaluate(model, df_test)

    # held_out_data = create_prediction_data(probs1)

    """transform big probs to be more similar to small probs"""

    #new_probs = trans0(probs1[0][2])

    """ The probability transformation function should have an argument stating which transformations should be
    carried out, and in which order. It should then, for each transformation, get the mean from the training data
    
    
    """

    # transformed_probs = probability_transformation(probs1, probs0)
    #
    # improvement_manhattan, improvement_weighted = compare_distributions(transformed_probs, probs0, probs1)
    #
    # print(f" Using the Manhattan distance, {improvement_manhattan * 100}% "
    #       f"of the transformed distributions are closer to the target distribution.\nFor the weighted Manhattan "
    #       f"distance, the score is {improvement_weighted * 100}%.")
