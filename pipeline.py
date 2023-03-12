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
from tqdm import tqdm
import pandas as pd

from GetClusters.differenceMetrics import bucket_diff_top_p, bucket_diff_top_k, compare_topk_success, compare_topk_prob
from GetClusters.differenceMetrics import compare_average_topk_len, entropy_difference
from GetClusters.featureVector import create_bucket_feature_vector, create_feature_vector, scale_feature_vector
from GetClusters.featureVector import create_and_save_scaled_feature_vector
from GetClusters.clustering import k_means_clustering
from GetClusters.clustering import find_optimal_n, label_distribution

sys.path.insert(1, '../Non-Residual-GANN')
from GenerateData.load_data_new import generateData
from Transformation.get_means_from_training_data import get_mean_entropy_from_training_data
from ClassifierFiles.classifier import BertClassifier
from ClassifierFiles.trainClassifier import train
from ClassifierFiles.evaluateClassifier import evaluate
from ClassifierFiles.trainingDataClassifier import prepare_training_data
from Transformation.transformation import classifyProbabilityIntoFamily, create_prediction_data, transformProbabilities
from Transformation.transformation import trans0
from Transformation.transformation import probability_transformation
from Transformation.transformation import compare_distributions


if __name__ == "__main__":

    """Load data"""
    NUM_TRAIN_SHEETS = 10000  # for creating feature vectors, and training classifier

    # small_probs, big_probs, small_indices_final, big_indices_final = generateData(
    #          num_samples=100000, truncate=True, topk=256, save=True)

    # for i in tqdm(range(9)):
    #     print("  => LOADING DATA")
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
    #     print("  => LOADING FEATURE VECTORS")
    #
    #     """create scaled feature vector"""
    #
    #     function_list = [bucket_diff_top_k, entropy_difference]  # difference metrics to use for creating feature vector
    #     create_and_save_scaled_feature_vector(function_list, probs0, probs1, NUM_TRAIN_SHEETS, i)

    scaled_features = np.zeros((64*NUM_TRAIN_SHEETS*9, 4))

    for i in range(9):
        print(f"OPENING file {i}")
        with open(f"train_data/scaled_features_{i}.pkl", "rb") as f:
            feature = pickle.load(f)
        print(f"DONE OPENING file {i}")
        scaled_features[i*(64*NUM_TRAIN_SHEETS):(i+1)*(64*NUM_TRAIN_SHEETS)] = feature

    print(f"Shape of scaled features: {scaled_features.shape}")
    print(f"[0] of scaled_features: {scaled_features[0]}")

        # with open(f"train_data/scaled_features.pkl", "rb") as k:
        #     scaled_features = pickle.load(k)

    # token_dict = pickle.load(open("reverseVocab.p", "rb"))
    # tokens = [token_dict[i] for i in range(len(token_dict))]
    #


    print("  => CLUSTERING")
    #print("  => FINDING OPTIMAL N")
    """find optimal number of clusters for clustering, and cluster"""
    # elbow = find_optimal_n(scaled_features)

    N_CLUSTERS = 3

    print("  => GET CLUSTER LABELS")

    labels = k_means_clustering(scaled_features, N_CLUSTERS)

    label_distribution = label_distribution(N_CLUSTERS, labels)

    print(f"label distribution: {label_distribution}")

    # print("  => GET MEAN ENTROPY")
    #
    # entropies = get_mean_entropy_from_training_data(NUM_TRAIN_SHEETS, probs0, labels, N_CLUSTERS)
    #
    # print(f"entropy means: {entropies}")
    # for i in range(len(entropies.keys())):
    #     print(f"mean entropy cluster {i}: {np.mean(entropies[f'entropy_{i}'])}")

    # data = create_feature_vector([bucket_diff_top_k], probs0, probs1, NUM_TRAIN_SHEETS)
    # mean_1 = -(np.mean(data[:, 0]))  # added minus because the bucket_diff_top_k returns how much more the big bucket
    #                                  # has than the small model
    # mean_2 = -(np.mean(data[:, 1]))
    # mean_3 = -(np.mean(data[:, 2]))
    #
    # print(mean_1)
    # print(mean_2)
    # print(mean_3)

    # feature_means = dict()
    # feature_means["entropy"] = [np.mean(entropies[f'entropy{i}']) for i in range(len(entropies.keys()))]
    #
    # print(feature_means["entropy"])

    """train classifier on big probs and labels from clustering step"""
    NUM_CLASSES = N_CLUSTERS
    BATCH_SIZE = 16
    EPOCHS = 5
    LR = 5e-7

    print("  => PREPARE TRAINING DATA FOR CLASSIFIER")

    df = False

    for i in tqdm(range(9)):
        with open(f"train_data/train_big_100000_{i}.pkl", "rb") as f:
            probs1 = pickle.load(f)
        probs1 = probs1.numpy()
        df_tmp = prepare_training_data(probs1, labels, NUM_TRAIN_SHEETS)
        if type(df) is bool:
            df = df_tmp
        else:
            df = pd.concat([df, df_tmp], axis=0, ignore_index=True)

    print(f"length of pandas frame: {df.shape[0]}")
    print(f"Number of columns: {len(df.columns)}")

    np.random.seed(112)
    torch.manual_seed(0)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                         [int(.8 * len(df)), int(.95 * len(df))])

    model = BertClassifier(NUM_CLASSES)

    print("  => STARTING TRAINING")
    train(model, df_train, df_val, LR, EPOCHS, BATCH_SIZE, num_classes=NUM_CLASSES)

    torch.save(model.state_dict(), f'first_try.pt')

    print("  => EVALUATING MODEL")
    """test classifier"""
    evaluate(model, df_test, num_classes=NUM_CLASSES)

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
