from scipy.stats import entropy
import numpy as np
import pickle
import sys
sys.path.insert(1, '../GetClusters')
import config
from GetClusters.featureVector import load_feature_vector
from GetClusters.differenceMetrics import get_entropy_feature
from GetClusters.differenceMetrics import bucket_diff_top_k

# TODO: the parameter associated with bucket_diff_top_k might change, this can't stay hard-coded!


def get_means_from_training_data(functions, num_features, num_sheets, labels=None):
    mean_dict = dict()

    feature_vector = load_feature_vector(functions=functions, num_features=num_features, num_sheets=num_sheets,
                                         scaled=False)  # num_sheets it the number of samples per file (usually 10000)

    if labels is not None:
        labels_array = np.array(labels)
        print(f"shape labels array: {labels_array.shape}")
        unique_labels = np.unique(labels_array)  # get number of unique elements in labels array

        completed_columns = 0
        columns_to_add = 0
        for function in functions:
            columns_to_add += config.function_feature_dict[f"{function.__name__}"]
            for label in unique_labels:  # loop with range number of unique elements
                bool_labels = labels_array == label  # first turn labels into booleans
                #features_for_boxplot = dict()
                for i in range(config.function_feature_dict[f"{function.__name__}"]):
                    mean_feature = np.mean(feature_vector[:, i+completed_columns], axis=0, where=bool_labels)  # then for each one, calculate the mean separately
                    # -------------- box plot -----------------------------------------------#
                    # features_tmp = feature_vector[:, i+completed_columns]
                    # features_for_boxplot[f"{i}_{function.__name__}"] = features_tmp[bool_labels]
                    mean_dict[f"{function.__name__}_{i}_{label}"] = mean_feature
            completed_columns += columns_to_add
            columns_to_add = 0
    else:
        for function in functions:
            for i in range(config.function_feature_dict[f"{function.__name__}"]):
                mean_feature = np.mean(feature_vector[:, i])
                mean_dict[f"{function.__name__}_{i}"] = mean_feature

    return mean_dict


if __name__ == "__main__":
    print(config.function_feature_dict)
    feature_vector = np.array([[[1, 3, 4, 5],
                               [6, 3, 6, 2],
                               [7, 6, 1, 4],
                               [3, 2, 4, 1]]])

    print(f"feature_vector shape: {feature_vector.shape}")
    labels_matrix = np.array([1,
                       0,
                       0,
                       1])

    function_list_1 = [bucket_diff_top_k, get_entropy_feature]

    dict_means = get_means_from_training_data(num_features=4, num_sheets=10000, functions=function_list_1,
                                              labels=labels_matrix)

    for key, value in dict_means.items():
        print(key, value)

