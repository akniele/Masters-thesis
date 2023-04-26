import sys
import pickle
import numpy as np

from GetClusters.clustering import k_means_clustering, label_distribution
from GetClusters.featureVector import load_feature_vector
from Transformation.get_means_from_training_data import get_means_from_training_data
from ClassifierFiles import train_and_evaluate_classifier
from ClassifierFiles.train_and_evaluate_classifier import make_predictions
sys.path.insert(1, '../Non-Residual-GANN')
from GenerateData.generate_data import generateData
from Transformation.transformation import transformations, get_distances


def train(function, bucket_indices, num_clusters, batch_size, epochs, lr, num_test_samples, num_features,
             generate_data=False, generate_sorted_by_big=False, train_classifier=False):

    NUM_TRAIN_SHEETS = 10_000  # for creating feature vectors, and training classifier
    N_CLUSTERS = num_clusters  # number of clusters used for clustering

    # for classifier:
    NUM_CLASSES = N_CLUSTERS  # number of classes for the classifier model (has to be same as number of clusters)
    BATCH_SIZE = batch_size
    EPOCHS = epochs
    LR = lr
    NUM_FEATURES = num_features

    if generate_data:
        print("  => GENERATING DATA AND FEATURE VECTORS")
        generateData(function, bucket_indices, num_features=NUM_FEATURES,
                     num_samples=100_000, truncate=True, topk=256, save=True,
                     features_only=True, sorted_by_big=False)

    print("  => LOADING SCALED FEATURES FOR CLUSTERING")
    scaled_features = load_feature_vector(function=function, num_features=NUM_FEATURES, num_sheets=NUM_TRAIN_SHEETS,
                                          scaled=True)

    print(f"Shape of scaled features: {scaled_features.shape}")

    print(scaled_features[:3])
    print(scaled_features[(NUM_TRAIN_SHEETS*64)-1:(NUM_TRAIN_SHEETS*64)+3])

    print("  => CLUSTERING")
    labels = k_means_clustering(scaled_features, N_CLUSTERS)

    labels_distribution = label_distribution(N_CLUSTERS, labels)
    print(f"label distribution: {labels_distribution}")

    print("  => GET MEAN FEATURES FROM TRAINING DATA")
    dict_means = get_means_from_training_data(function=function, num_features=NUM_FEATURES,
                                              num_sheets=NUM_TRAIN_SHEETS, labels=labels)

    for key, value in dict_means.items():
        print(f"key: {key}\t value:{value}")

    if train_classifier:
        pred_labels, true_labels = train_and_evaluate_classifier.train_and_evaluate_classifier(
            NUM_CLASSES, BATCH_SIZE, EPOCHS, LR, labels, NUM_TRAIN_SHEETS, function)

    new_pred_labels = make_predictions(num_classes=NUM_CLASSES, num_sheets=num_test_samples)

    print(f"pred labels: {new_pred_labels[:50]}")

    new_pred_labels = np.reshape(new_pred_labels[:num_test_samples*64], (num_test_samples, 64))

    print(f"new shape of pred labels: {new_pred_labels.shape}")

    return new_pred_labels, dict_means


def load_test_data(new_pred_labels, num_test_samples, bucket_indices):
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

    with open(f"train_data/indices_small_100000_9.pkl", "rb") as g:
        indices0 = pickle.load(g)

    indices0 = indices0[:num_test_samples].numpy()

    bucket_indices.insert(0, 0)
    bucket_indices.append(16_384)

    return bigprobs, smallprobs, indices1, indices0, new_pred_labels, num_test_samples


def transform_and_evaluate(bigprobs, smallprobs, indices1, indices0, dict_means,
                           bucket_indices, function, new_pred_labels, num_features):

    transformed_probs, original_probs = transformations(bigprobs,
                                                        indices1,
                                                        dict_means,
                                                        num_features,
                                                        bucket_indices,
                                                        function,
                                                        upper_bound=130,
                                                        top_p=0.9,
                                                        pred_labels=new_pred_labels)

    print(f"shape transformed_probs: {transformed_probs.shape}")
    print(f" example of transformed probs: {transformed_probs[0][0][:30]}")

    trans_distances_tmp, original_distances_tmp = get_distances(transformed_probs, original_probs,
                                                                smallprobs,
                                                                indices0)

    trans_distances_tmp = np.expand_dims(trans_distances_tmp, -1)
    original_distances_tmp = np.expand_dims(original_distances_tmp, -1)

    return trans_distances_tmp, original_distances_tmp


def generate_data(function, bucket_indices, num_samples=100_000, truncate=True, topk=256, save=True):
    print("  => GENERATING DATA AND FEATURE VECTORS")
    generateData(function, bucket_indices, num_samples=num_samples, truncate=truncate, topk=topk, save=save)
