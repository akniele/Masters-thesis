import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle


def scale_feature_vector(feature_vector):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_vector)
    return scaled_features


def load_feature_vector(function, bucket_indices, top_p, num_features, num_sheets=10000, scaled=True):
    features = np.zeros((64*num_sheets*9, num_features))

    if function.__name__ == "get_entropy_feature":
        filename = f"{function.__name__}"
    elif function.__name__ == "bucket_diff_top_k":
        filename = f"{function.__name__}_{'-'.join([str(i) for i in bucket_indices])}"
    elif function.__name__ == "get_top_p_difference":
        filename = f"{function.__name__}_{top_p}"
    else:
        raise Exception(f"{function.__name__} is not a valid transformation function.")

    for i in range(9):
        with open(f"train_data/features_{i}_{filename}.pkl", "rb") as f:
            feature = pickle.load(f)
            feature = feature.numpy()

        correct_shape_feature = feature.reshape((feature.shape[0]*feature.shape[1], num_features))
        features[i*(64*num_sheets):(i+1)*(64*num_sheets)] = correct_shape_feature

    if scaled:
        features = scale_feature_vector(features)
    return features
