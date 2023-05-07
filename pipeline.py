import sys
import pickle
import numpy as np
from scipy.stats import entropy

from GetClusters.clustering import k_means_clustering, label_distribution
from GetClusters.featureVector import load_feature_vector
from Transformation.get_means_from_training_data import get_means_from_training_data
from ClassifierFiles import train_and_evaluate_classifier
from ClassifierFiles.train_and_evaluate_classifier import make_predictions
sys.path.insert(1, '../Non-Residual-GANN')
from GenerateData.generate_data import generateData
from Transformation.transformation import transformations, get_distances
import time
from Transformation.transformation import get_mean_distances
from Transformation.histogram_of_distances import difference_histogram
from Transformation.fill_up_distributions import fill_multiple_distributions
from baseline_model import baseline
import os
import torch


def run_baseline(n_test_samples, batch_size, epochs, lr, generate_data=False, generate_sorted_by_big_data=False):
    filename = f"baseline_{batch_size}_{epochs}_{lr}"
    with open(f"logfiles/{filename}.txt", "w") as f:
        f.write(f"Log file baseline\n"
                f"Batch size: {batch_size}\n"
                f"Number of epochs: {epochs}\n"
                f"Learning rate: {lr}\n")
    if generate_data:
        print("  => GENERATING DATA AND FEATURE VECTORS")
        generateData(function=None, bucket_indices=None, top_p=None, num_features=None,
                     num_samples=100_000, truncate=True, topk=256, save=True, sorted_by_big=False)

    if generate_sorted_by_big_data:
        print("  => GENERATING DATA SORTED BY BIG MODEL")
        generateData(function=None, bucket_indices=None, top_p=None, num_features=None,
                     num_samples=100_000, truncate=True, topk=256, save=True, sorted_by_big=True)

    start = time.perf_counter()
    baseline(n_test_samples, batch_size, epochs, lr, filename=filename)
    end = time.perf_counter()
    with open(f"logfiles/{filename}.txt", "a") as f:
        f.write(f"elapsed time:{(end - start) / 60} minutes\n")


def run_transparent_pipeline(function, n_clusters, batch_size, epochs, lr, generate_data, generate_sorted_by_big,
                             train_classifier, bucket_indices, top_p, n_test_samples):

    # ----- make directories for log files, plots and models if they don't already exist -------- #

    logfile_path = '/home/ubuntu/pipeline/logfiles'  # create new directory for log files if it doesn't already exist
    directory_exists = os.path.exists(logfile_path)
    if not directory_exists:
        os.makedirs(logfile_path)

    plot_path = '/home/ubuntu/pipeline/plots'  # create new directory for plots if it doesn't already exist
    directory_exists = os.path.exists(plot_path)
    if not directory_exists:
        os.makedirs(plot_path)

    model_path = '/home/ubuntu/pipeline/models'  # create new directory for models if it doesn't already exist
    directory_exists = os.path.exists(model_path)
    if not directory_exists:
        os.makedirs(model_path)

    data_path = '/home/ubuntu/pipeline/train_data'  # create new directory for train data if it doesn't already exist
    directory_exists = os.path.exists(data_path)
    if not directory_exists:
        os.makedirs(data_path)

    # ----- done making directories ------------------------------------------------------------- #

    if function.__name__ == "get_entropy_feature":
        num_features = 1
        filename = f"{function.__name__}_{n_clusters}_{batch_size}_{epochs}_{lr}"
        with open(f"logfiles/{filename}.txt", "w") as f:
            f.write(f"Log file\n"
                    f"Transformation function: {function.__name__}\n"
                    f"Number of clusters: {n_clusters}\n"
                    f"Batch size: {batch_size}\n"
                    f"Number of epochs: {epochs}\n"
                    f"Learning rate: {lr}\n"
                    f"Generate training data: {generate_data}\n"
                    f"Generate training data sorted by big model: {generate_sorted_by_big}\n"
                    f"Train classifier: {train_classifier}\n")
            f.close()

    elif function.__name__ == "bucket_diff_top_k":
        num_features = len(bucket_indices) + 1
        filename = f"{function.__name__}_{'-'.join([str(i) for i in bucket_indices])}_{n_clusters}_{batch_size}_{epochs}_{lr}"
        with open(f"logfiles/{filename}.txt", "w") as f:
            f.write(f"Log file\n"
                    f"Transformation function: {function.__name__}\n"
                    f"Bucket indices: {bucket_indices}\n"
                    f"Number of clusters: {n_clusters}\n"
                    f"Batch size: {batch_size}\n"
                    f"Number of epochs: {epochs}\n"
                    f"Learning rate: {lr}\n"
                    f"Generate training data: {generate_data}\n"
                    f"Generate training data sorted by big model: {generate_sorted_by_big}\n"
                    f"Train classifier: {train_classifier}")
            f.close()

    elif function.__name__ == "get_top_p_difference":
        num_features = 1
        filename = f"{function.__name__}_{top_p}_{n_clusters}_{batch_size}_{epochs}_{lr}"
        with open(f"logfiles/{filename}.txt", "w") as f:
            f.write(f"Log file\n"
                    f"Transformation function: {function.__name__}\n"
                    f"Top-p: {top_p}\n"
                    f"Number of clusters: {n_clusters}\n"
                    f"Batch size: {batch_size}\n"
                    f"Number of epochs: {epochs}\n"
                    f"Learning rate: {lr}\n"
                    f"Generate training data: {generate_data}\n"
                    f"Generate training data sorted by big model: {generate_sorted_by_big}\n"
                    f"Train classifier: {train_classifier}\n")
            f.close()

    else:
        raise Exception(f"{function.__name__} is not a valid transformation function.")

    start = time.perf_counter()

    new_pred_labels, dict_means = train(function=function,
                                        bucket_indices=bucket_indices,
                                        top_p=top_p,
                                        num_clusters=n_clusters,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        lr=lr,
                                        num_test_samples=n_test_samples,
                                        num_features=num_features,
                                        filename=filename,
                                        generate_data=generate_data,
                                        generate_sorted_by_big_data=generate_sorted_by_big,
                                        train_classifier=train_classifier)

    bigprobs, smallprobs, indices1, indices0 = load_test_data(num_test_samples=n_test_samples,
                                                              bucket_indices=bucket_indices)

    trans_distances, original_distances = transform_and_evaluate(bigprobs,
                                                                 smallprobs,
                                                                 indices1,
                                                                 indices0,
                                                                 dict_means,
                                                                 filename,
                                                                 bucket_indices=bucket_indices,
                                                                 top_p=top_p,
                                                                 function=function,
                                                                 new_pred_labels=new_pred_labels,
                                                                 num_features=num_features,
                                                                 n_test_samples=n_test_samples)

    difference_histogram(trans_distances, original_distances, filename)

    score_mean, score_std = get_mean_distances(trans_distances, original_distances, filename)
    end = time.perf_counter()

    with open(f"logfiles/{filename}.txt", "a") as f:
        f.write(f"Difference in mean Weighted Manhattan Distance: {score_mean}\n"
                f"Difference in standard deviation of Weighted Manhattan Distances: {score_std}\n"
                f"elapsed time:{(end - start) / 60} minutes\n")


def train(function, bucket_indices, top_p, num_clusters, batch_size, epochs, lr, num_test_samples, num_features,
          filename, generate_data=False, generate_sorted_by_big_data=False, train_classifier=False):

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
        generateData(function, bucket_indices, top_p, num_features=NUM_FEATURES,
                     num_samples=100_000, truncate=True, topk=256, save=True, sorted_by_big=False)

    if generate_sorted_by_big_data:
        print("  => GENERATING DATA SORTED BY BIG MODEL")
        generateData(function, bucket_indices, top_p, num_features=NUM_FEATURES,
                     num_samples=100_000, truncate=True, topk=256, save=True, sorted_by_big=True)

    print("  => LOADING SCALED FEATURES FOR CLUSTERING")
    scaled_features = load_feature_vector(function=function, bucket_indices=bucket_indices, top_p=top_p,
                                          num_features=NUM_FEATURES, num_sheets=NUM_TRAIN_SHEETS, scaled=True)

    if N_CLUSTERS is not None:
        print("  => CLUSTERING")
        labels = k_means_clustering(scaled_features, N_CLUSTERS)

        labels_distribution = label_distribution(N_CLUSTERS, labels)
        with open(f"logfiles/{filename}.txt", "a") as logfile:
            logfile.write(f"Label distribution of clustering: {labels_distribution}\n")
    else:
        labels = None

    print("  => GET MEAN FEATURES FROM TRAINING DATA")
    dict_means = get_means_from_training_data(function=function, bucket_indices=bucket_indices, top_p=top_p,
                                              num_features=NUM_FEATURES, num_sheets=NUM_TRAIN_SHEETS, labels=labels)

    with open(f"logfiles/{filename}.txt", "a") as logfile:
        logfile.write("Mean feature differences from training data:\n")
        for key, value in dict_means.items():
            logfile.write(f"Feature: {key}\t Mean difference: {value}\n")

    if labels is not None:
        if train_classifier:
            pred_labels, true_labels = train_and_evaluate_classifier.train_and_evaluate_classifier(
                NUM_CLASSES, BATCH_SIZE, EPOCHS, LR, labels, NUM_TRAIN_SHEETS, function, filename)

        new_pred_labels = make_predictions(num_classes=NUM_CLASSES, num_sheets=num_test_samples, function=function,
                                           epochs=EPOCHS, lr=LR)

        print(f"pred labels: {new_pred_labels[:50]}")

        new_pred_labels = np.reshape(new_pred_labels[:num_test_samples*64], (num_test_samples, 64))

    else:
        new_pred_labels = None

    return new_pred_labels, dict_means


def load_test_data(num_test_samples, bucket_indices):
    print("  => LOAD DATA FOR TRANSFORMATION")

    with open(f"train_data/big_10000_9.pkl", "rb") as f:
        bigprobs = pickle.load(f)  # float32

    bigprobs = bigprobs.to(torch.float64)

    bigprobs += 10e-10  # get rid of zeros by adding this very small value to each probability

    bigprobs = bigprobs[:num_test_samples].numpy()  # still float32

    with open(f"train_data/indices_big_10000_9.pkl", "rb") as g:
        indices1 = pickle.load(g)

    indices1 = indices1[:num_test_samples].numpy()

    print("  => LOAD DATA FOR EVALUATION")

    with open(f"train_data/small_10000_9.pkl", "rb") as f:
        smallprobs = pickle.load(f)

    smallprobs = smallprobs.to(torch.float64)
    smallprobs += 10e-10
    smallprobs = smallprobs[:num_test_samples].numpy()

    with open(f"train_data/indices_small_10000_9.pkl", "rb") as g:
        indices0 = pickle.load(g)

    indices0 = indices0[:num_test_samples].numpy()

    bucket_indices.insert(0, 0)
    bucket_indices.append(16_384)

    return bigprobs, smallprobs, indices1, indices0


def transform_and_evaluate(bigprobs, smallprobs, indices1, indices0, dict_means, filename,
                           bucket_indices, top_p, function, new_pred_labels, num_features, n_test_samples):
    epsilon = 10e-10
    bigprobs += epsilon
    smallprobs += epsilon

    transformed_probs, original_probs = transformations(bigprobs,
                                                        indices1,
                                                        dict_means,
                                                        num_features,
                                                        bucket_indices,
                                                        function,
                                                        upper_bound=141.356,
                                                        top_p=top_p,
                                                        pred_labels=new_pred_labels)

    print(f"shape transformed_probs: {transformed_probs.shape}")
    print(f"number of zeros in array: {(transformed_probs.size - np.count_nonzero(transformed_probs))}")
    print(f"number of negative values in array: {np.sum(transformed_probs < 0)}")

    transformed_probs += epsilon  # to get rid of any potential 0s in the distributions
    original_probs += epsilon

    print("added epsilon")

    filled_up_small_probs = fill_multiple_distributions(smallprobs, indices0)
    print("filled up small probs")
    filled_up_small_probs += epsilon

    print("calculating entropy")

    small_entropy = np.mean(entropy(filled_up_small_probs, axis=-1))
    print("second one")
    transformed_entropy = np.mean(entropy(transformed_probs, axis=-1))
    print("last one")
    original_entropy = np.mean(entropy(original_probs, axis=-1))

    with open(f"logfiles/{filename}.txt", "a") as logfile:
        logfile.write(f"mean entropy of transformed distributions: {transformed_entropy}\n"
                      f"mean entropy of original big distributions: {original_entropy}\n"
                      f"mean entropy of original small distributions: {small_entropy}\n")

    del small_entropy
    del transformed_entropy
    del original_entropy

    print("index")
    index_highest_prob_big = np.argmax(original_probs, axis=-1)
    print("index 2")
    index_highest_prob_trans = np.argmax(transformed_probs, axis=-1)
    print("index 3")
    index_highest_prob_small = np.argmax(filled_up_small_probs, axis=-1)

    print("acc")
    accuracy_trans_tmp = index_highest_prob_trans == index_highest_prob_small
    print("acc 2")
    accuracy_orig_tmp = index_highest_prob_big == index_highest_prob_small

    print("calculate something")
    accuracy_trans = np.count_nonzero(accuracy_trans_tmp) / (n_test_samples * 64)
    print("one more time")
    accuracy_orig = np.count_nonzero(accuracy_orig_tmp) / (n_test_samples * 64)

    with open(f"logfiles/{filename}.txt", "a") as logfile:
        logfile.write(f"Percentage of true tokens that had the highest probability in the original "
                      f"distributions: {accuracy_orig}%\n"
                      f"Percentage of true tokens that had the highest probability in the transformed "
                      f"distributions: {accuracy_trans}%\n")

    del index_highest_prob_trans, index_highest_prob_big, index_highest_prob_small
    del accuracy_trans_tmp, accuracy_orig_tmp
    del accuracy_trans, accuracy_orig

    print("trans dist")
    trans_distances_tmp, original_distances_tmp = get_distances(transformed_probs, original_probs,
                                                                filled_up_small_probs)

    print("expand dims")
    trans_distances_tmp = np.expand_dims(trans_distances_tmp, -1)
    original_distances_tmp = np.expand_dims(original_distances_tmp, -1)

    return trans_distances_tmp, original_distances_tmp
