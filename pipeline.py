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


from GetClusters.clustering import k_means_clustering
from GetClusters.clustering import find_optimal_n, label_distribution
from GetClusters.differenceMetrics import entropy_difference
from GetClusters.featureVector import load_scaled_feature_vector
from GetClusters.featureVector import create_and_save_scaled_feature_vector
from ClassifierFiles import train_and_evaluate_classifier

sys.path.insert(1, '../Non-Residual-GANN')
# from GenerateData.load_data_new import generateData
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

    # bucket_diff_top_k,
    # function_list = [entropy_difference]  # difference metrics to use for creating feature vector
    #
    # create_and_save_scaled_feature_vector(function_list, NUM_TRAIN_SHEETS)

    # --------------------------------------------------------------------------------------#
    scaled_features = load_scaled_feature_vector(NUM_TRAIN_SHEETS, num_features=4)

    print(f"Shape of scaled features: {scaled_features.shape}")
    print(f"[0] of scaled_features: {scaled_features[0]}")

    print(scaled_features[:3])
    print(scaled_features[(NUM_TRAIN_SHEETS*64)-1:(NUM_TRAIN_SHEETS*64)+3])

    # print(len(np.unique(scaled_features))/4)
    # print(len(scaled_features))
    # # check for duplicates (if True, there are duplicates)
    # print((len(np.unique(scaled_features))/4) < len(scaled_features))
    # ---------------------------------------------------------------------------------------#


    # token_dict = pickle.load(open("reverseVocab.p", "rb"))
    # tokens = [token_dict[i] for i in range(len(token_dict))]

    # elbow = find_optimal_n(scaled_features)

    # ---------------------------------------------------------------------#
    #
    N_CLUSTERS = 3

    labels = k_means_clustering(scaled_features, N_CLUSTERS)

    label_distribution = label_distribution(N_CLUSTERS, labels)

    print(f"label distribution: {label_distribution}")


    # ------------------------------------------------------------------------#

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

    # -------------------------------------------------------------------------------#

    NUM_CLASSES = 3
    BATCH_SIZE = 16
    EPOCHS = 100
    LR = 5e-5

    pred_labels, true_labels = train_and_evaluate_classifier.train_and_evaluate_classifier(
        NUM_CLASSES, BATCH_SIZE, EPOCHS, LR, labels, NUM_TRAIN_SHEETS)

    # ------------------------------------------------------------------------------------#


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
