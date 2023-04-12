from GetClusters.differenceMetrics import bucket_diff_top_k, get_entropy_feature
from pipeline import train, test, generate_data
import time
from multiprocess import Process, Queue
import torch
import numpy as np
from scipy.stats import entropy

from Transformation.transformation import transformations, get_distances, get_mean_distances


def transform_and_evaluate(bigprobs, smallprobs, indices1, indices0, dict_means,
                           bucket_indices, functions, num_test_samples, new_pred_labels, samp_per_iter, q, i):
    print(f"transforming probs with indices {i*samp_per_iter} until {(i+1)*samp_per_iter}")
    print(f"entropy first distribution before transformations: {entropy(bigprobs[i*samp_per_iter][0])}")
    transformed_probs, original_probs = transformations(bigprobs[i * samp_per_iter:(i + 1) * samp_per_iter],
                                                        indices1[i * samp_per_iter:(i + 1) * samp_per_iter],
                                                        dict_means, bucket_indices, functions, num_test_samples,
                                                        upper_bound=130,
                                                        pred_labels=new_pred_labels[i * samp_per_iter:(i + 1) * samp_per_iter])

    print(f"entropy first distribution after transformations: {entropy(transformed_probs[i * samp_per_iter][0])}")
    print(f"shape transformed_probs: {transformed_probs.shape}")
    print(f" example of transformed probs: {transformed_probs[0][0][:30]}")

    trans_distances_tmp, original_distances_tmp = get_distances(transformed_probs, original_probs,
                                                                smallprobs[i * samp_per_iter:(i + 1) * samp_per_iter],
                                                                indices0[i * samp_per_iter:(i + 1) * samp_per_iter])

    trans_distances_tmp = np.expand_dims(trans_distances_tmp, -1)
    original_distances_tmp = np.expand_dims(original_distances_tmp, -1)
    print(f" value of i to be put in queue: {i}")
    q.put((trans_distances_tmp, original_distances_tmp, i))


# def transform_and_evaluate(bigprobs, smallprobs, indices1, indices0, dict_means,
#                            bucket_indices, functions, num_test_samples, new_pred_labels):
#     transformed_probs, original_probs = transformations(bigprobs,
#                                                         indices1,
#                                                         dict_means, bucket_indices, functions, num_test_samples,
#                                                         upper_bound=130,
#                                                         pred_labels=new_pred_labels)
#
#     print(f"shape transformed_probs: {transformed_probs.shape}")
#     print(f" example of transformed probs: {transformed_probs[0][0][:30]}")
#
#     trans_distances_tmp, original_distances_tmp = get_distances(transformed_probs, original_probs,
#                                                                 smallprobs,
#                                                                 indices0)
#
#     trans_distances_tmp = np.expand_dims(trans_distances_tmp, -1)
#     original_distances_tmp = np.expand_dims(original_distances_tmp, -1)
#
#     return trans_distances_tmp, original_distances_tmp

    #q.put((trans_distances_tmp, original_distances_tmp, i))


def multi_process(bigprobs, smallprobs, indices1, indices0, new_pred_labels, num_test_samples, dict_means,
                  bucket_indices, functions, range_multi_processing, samp_per_iter):
    q = Queue()
    processes = []
    trans_distances = np.zeros((num_test_samples, 64, 1))
    original_distances = np.zeros((num_test_samples, 64, 1))
    for i in range(range_multi_processing):
        p = Process(target=transform_and_evaluate, args=(bigprobs, smallprobs, indices1,
                                                         indices0, dict_means, bucket_indices, functions,
                                                         num_test_samples, new_pred_labels, samp_per_iter, q, i))
        processes.append(p)
        p.start()
    for p in processes:
        ret = q.get()  # will block
        trans_distances[ret[2] * samp_per_iter:(ret[2] + 1) * samp_per_iter] = ret[0]
        original_distances[ret[2] * samp_per_iter:(ret[2] + 1) * samp_per_iter] = ret[1]
    for p in processes:
        p.join()

    return trans_distances, original_distances


if __name__ == "__main__":
    BUCKET_INDICES: list = [10, 35]
    FUNCTION = [bucket_diff_top_k, get_entropy_feature]
    N_CLUSTERS: int = 3

    BATCH_SIZE: int = 16
    EPOCHS: int = 50
    LR: float = 5e-5

    N_TEST_SAMPLES = 6

    samp = 2

    GENERATE_DATA: bool = False
    TRAIN_CLASSIFIER: bool = False

    RANGE_MULTI_PROCESSING = N_TEST_SAMPLES//samp

    start = time.perf_counter()

    new_pred_labels, dict_means = train(functions=FUNCTIONS, bucket_indices=BUCKET_INDICES,
                     num_clusters=N_CLUSTERS, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR,
                     num_test_samples=N_TEST_SAMPLES, generate_data=GENERATE_DATA, train_classifier=TRAIN_CLASSIFIER)

    bigprobs, smallprobs, indices1, indices0, new_pred_labels, num_test_samples = test(new_pred_labels,
                                                                                       dict_means,
                                                                                       num_test_samples=N_TEST_SAMPLES,
                                                                                       bucket_indices=BUCKET_INDICES,
                                                                                       functions=FUNCTIONS)



    trans_distances, original_distances = multi_process(bigprobs, smallprobs, indices1, indices0,
                                                       new_pred_labels, num_test_samples, dict_means,
                                                       bucket_indices=BUCKET_INDICES, functions=FUNCTIONS,
                                                       range_multi_processing=RANGE_MULTI_PROCESSING,
                                                        samp_per_iter=samp)

    # trans_distances, original_distances = transform_and_evaluate(bigprobs, smallprobs, indices1,
    #                                                              indices0, dict_means, bucket_indices=BUCKET_INDICES,
    #                                                              functions=FUNCTIONS, num_test_samples=N_TEST_SAMPLES,
    #                                                              new_pred_labels=new_pred_labels)

    print(f"shape trans_distances: {trans_distances.shape}")
    print(f"original_distances: {original_distances}")

    score = get_mean_distances(trans_distances, original_distances)

    print(f"final score: {score}")

    end = time.perf_counter()

    print(f"elapsed time:{(end - start) / 60} minutes")
