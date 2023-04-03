from GetClusters.differenceMetrics import bucket_diff_top_k, get_entropy_feature
from pipeline import pipeline, generate_data
import time
from multiprocess import Process, Queue


def multi_add():  # spawns child processes
    q = Queue()
    processes = []
    rets = []
    for i in range(0, 10):
        p = Process(target=do_something, args=(i, q))
        processes.append(p)
        p.start()
    for p in processes:
        ret = q.get()  # will block
        rets.append(ret)
    for p in processes:
        p.join()
    return rets


if __name__ == "__main__":
    BUCKET_INDICES: list = [10, 35]
    FUNCTIONS: list = [bucket_diff_top_k, get_entropy_feature]
    N_CLUSTERS: int = 3

    BATCH_SIZE: int = 16
    EPOCHS: int = 50
    LR: float = 5e-5

    N_TEST_SAMPLES = 100

    GENERATE_DATA: bool = False
    TRAIN_CLASSIFIER: bool = False

    start = time.perf_counter()

    score = pipeline(functions=FUNCTIONS, bucket_indices=BUCKET_INDICES,
                     num_clusters=N_CLUSTERS, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR,
                     num_test_samples=N_TEST_SAMPLES, generate_data=GENERATE_DATA, train_classifier=TRAIN_CLASSIFIER)

    print(f"final score: {score}")

    end = time.perf_counter()

    print(f"elapsed time:{(end - start) / 60} minutes")
