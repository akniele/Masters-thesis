import sys
sys.path.insert(1, '../GetClusters')
from GetClusters.featureVector import load_feature_vector
import numpy as np


def get_means_from_training_data(function, num_features, num_sheets, labels=None):
    mean_dict = dict()

    feature_vector = load_feature_vector(function=function, num_features=num_features, num_sheets=num_sheets,
                                         scaled=False)  # num_sheets it the number of samples per file (usually 10000)

    if labels is not None:
        labels_array = np.array(labels)
        print(f"shape labels array: {labels_array.shape}")
        unique_labels = np.unique(labels_array)  # get number of unique elements in labels array

        for label in unique_labels:  # loop with range number of unique elements
            bool_labels = labels_array == label  # first turn labels into booleans
            for i in range(num_features):
                mean_feature = np.mean(feature_vector[:, i], axis=0, where=bool_labels)  # then for each one, calculate the mean separately
                mean_dict[f"{function.__name__}_{i}_{label}"] = mean_feature

    else:
        for i in range(num_features):
            mean_feature = np.mean(feature_vector[:, i])
            mean_dict[f"{function.__name__}_{i}"] = mean_feature

    return mean_dict
