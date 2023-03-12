"""Create feature vector for clustering"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm


def create_bucket_feature_vector(functions, bucket_list, num_sheets):
    feature_vector = []
    for i, sheet in enumerate(probs0[:num_sheets]):  # for sheet in sheets
        for j in range(len(sheet)):  # for sample in sheet
            vector = []
            for function in functions: # right now there is only one function
                for k, bucket in enumerate(bucket_list):  # loop over list with the different indices and numbers of buckets
                    number_of_buckets, indices = bucket_list[k]
                    difference_metric = function(probs0[i][j], probs1[i][j], number_of_buckets, indices)
                    vector.extend(difference_metric)
            feature_vector.append(vector)
    return np.array(feature_vector, dtype=float)


def create_feature_vector(functions, probs0, probs1, num_sheets):
    feature_vector = []
    for i, sheet in enumerate(tqdm(probs0[:num_sheets], desc="creating feature vector")):  # for sheet in sheets
        for j in range(len(sheet)):  # for sample in sheet
            vector = []
            for function in functions:
                difference_metric = function(probs0[i][j], probs1[i][j])
                vector.extend(difference_metric)
            feature_vector.append(vector)
    return np.array(feature_vector, dtype=float)


def scale_feature_vector(feature_vector):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_vector)
    return scaled_features


def create_and_save_scaled_feature_vector(functions, probs0, probs1, num_sheets, index):
    feature_vector = create_feature_vector(functions, probs0, probs1, num_sheets)
    scaled_feature_vector = scale_feature_vector(feature_vector)
    with open(f"train_data/scaled_features_{index}.pkl", "wb") as h:
        pickle.dump(scaled_feature_vector, h)