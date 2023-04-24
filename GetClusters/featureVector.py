"""Create feature vector for clustering"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import config
from GetClusters.differenceMetrics import get_entropy_feature
from GetClusters.differenceMetrics import bucket_diff_top_k


def scale_feature_vector(feature_vector):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_vector)
    return scaled_features


def load_feature_vector(functions, num_features, num_sheets=10000, scaled=True):  # num features depends on the number of feature functions
    # used, and on how many features they each have
    features = np.zeros((64*num_sheets*9, num_features))

    #function_names = "_".join([function.__name__ for function in functions])

    for i in range(9):
        with open(f"train_data/features_{i}_{functions[0].__name__}.pkl", "rb") as f:
            feature = pickle.load(f)

        correct_shape_feature = feature.reshape((feature.shape[0]*feature.shape[1], num_features))
        features[i*(64*num_sheets):(i+1)*(64*num_sheets)] = correct_shape_feature

    if scaled:
        features = scale_feature_vector(features)
    return features


if __name__ == "__main__":
    functions = [get_entropy_feature, bucket_diff_top_k]
    num_features = sum([config.function_feature_dict[f"{function.__name__}"] for function in functions])
    print(num_features)

