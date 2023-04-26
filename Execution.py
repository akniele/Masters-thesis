from GetClusters.differenceMetrics import bucket_diff_top_k
from GetClusters.differenceMetrics import get_entropy_feature
from GetClusters.differenceMetrics import get_top_p_difference
from pipeline import train, load_test_data, transform_and_evaluate
import time
from Transformation.transformation import get_mean_distances


if __name__ == "__main__":
    # -------- hyperparameters -------- #
    function = get_top_p_difference  # get_entropy_feature  # bucket_diff_top_k
    BUCKET_INDICES = [10, 35]        # if bucket_diff_top_k, choose where buckets should start and end,
    # NB: 0 and len(distribution) are added automatically later!
    N_CLUSTERS = 3                   # number of clusters to use for k-means clustering
    BATCH_SIZE = 16
    EPOCHS = 50
    LR = 5e-5
    GENERATE_DATA = True             # if True, generates probability distributions as training data, and features
    TRAIN_CLASSIFIER = True          # if True, trains a new classifier
    N_TEST_SAMPLES = 500             # number of samples used for testing,
    # each sample consists of a sequence of 64 distributions
    # ----- end of hyperparameters ----- #

    if function.__name__ == "get_entropy_feature":
        NUM_FEATURES = [1]

    elif function.__name__ == "bucket_diff_top_k":
        NUM_FEATURES = [len(BUCKET_INDICES) + 1]

    elif function.__name__ == "get_top_p_difference":
        NUM_FEATURES = [1]

    else:
        raise Exception(f"{function.__name__} is not a valid transformation function.")

    start = time.perf_counter()

    new_pred_labels, dict_means = train(function=function,
                                        bucket_indices=BUCKET_INDICES,
                                        num_clusters=N_CLUSTERS,
                                        batch_size=BATCH_SIZE,
                                        epochs=EPOCHS,
                                        lr=LR,
                                        num_test_samples=N_TEST_SAMPLES,
                                        num_features=NUM_FEATURES,
                                        generate_data=GENERATE_DATA,
                                        train_classifier=TRAIN_CLASSIFIER)

    bigprobs, smallprobs, indices1, indices0, new_pred_labels, num_test_samples = load_test_data(new_pred_labels,
                                                                                       num_test_samples=N_TEST_SAMPLES,
                                                                                       bucket_indices=BUCKET_INDICES)

    trans_distances, original_distances = transform_and_evaluate(bigprobs,
                                                                 smallprobs,
                                                                 indices1,
                                                                 indices0,
                                                                 dict_means,
                                                                 bucket_indices=BUCKET_INDICES,
                                                                 function=function,
                                                                 new_pred_labels=new_pred_labels,
                                                                 num_features=NUM_FEATURES)

    print(f"shape trans_distances: {trans_distances.shape}")
    print(f"original_distances: {original_distances.shape}")

    score = get_mean_distances(trans_distances, original_distances)

    print(f"final score: {score}")

    end = time.perf_counter()

    print(f"elapsed time:{(end - start) / 60} minutes")
