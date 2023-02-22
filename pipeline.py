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

from differenceMetrics import bucket_diff_top_p, bucket_diff_top_k, compare_topk_success, compare_topk_prob
from differenceMetrics import compare_average_topk_len, entropy_difference
from featureVector import create_bucket_feature_vector, create_feature_vector, scale_feature_vector
from clustering import k_means_clustering
from clustering import find_optimal_n, label_distribution
from classifier import BertClassifier
from trainClassifier import train
from evaluateClassifier import evaluate
from trainingDataClassifier import prepare_training_data
from transformation import classifyProbabilityIntoFamily, create_prediction_data, transformProbabilities
from transformation import trans0


if __name__ == "__main__":

    """Load data"""
    # data
    probs0 = pickle.load(open("probs_0.p", "rb"))
    probs1 = pickle.load(open("probs_1.p", "rb"))
    probs0 = probs0.detach().numpy()
    probs1 = probs1.detach().numpy()
    token_dict = pickle.load(open("reverseVocab.p", "rb"))
    tokens = [token_dict[i] for i in range(len(token_dict))]

    """create scaled feature vector"""
    function_list = [bucket_diff_top_k]  # difference metrics to use for creating feature vector
    bucket_list = [(3, [0, 10, 35]), (4, [0, 5, 15, 45]), (2, [0, 20])]
    feature_vector = create_bucket_feature_vector(function_list, bucket_list)
    #feature_vector = create_feature_vector(function_list)

    scaled_features = scale_feature_vector(feature_vector)  # scale feature vector

    """find optimal number of clusters for clustering, and cluster"""
    elbow, silhouette, calinski = find_optimal_n(scaled_features)

    N_CLUSTERS = calinski

    labels = k_means_clustering(scaled_features, N_CLUSTERS)

    """train classifier on big probs and labels from clustering step"""
    NUM_CLASSES = N_CLUSTERS
    BATCH_SIZE = 16
    EPOCHS = 20
    LR = 5e-5

    df = prepare_training_data(probs1, labels)

    np.random.seed(112)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                         [int(.8 * len(df)), int(.95 * len(df))])

    model = BertClassifier(NUM_CLASSES)

    train(model, df_train, df_val, LR, EPOCHS, BATCH_SIZE)

    torch.save(model.state_dict(), f'first_try.pt')

    """test classifier"""
    evaluate(model, df_test)

    """transform big probs to be more similar to small probs"""

    held_out_data = create_prediction_data(probs1)
    transformed_probs = transformProbabilities(held_out_data)
    # TODO: adapt comparison function to compare all distributions at once
    # TODO: write function that changes the entropy of a distribution
    # TODO: debug this code section by section
