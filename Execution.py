from GetClusters.differenceMetrics import bucket_diff_top_k, get_entropy_feature
from pipeline import pipeline, generate_data


if __name__ == "__main__":
    BUCKET_INDICES = [10, 35]
    FUNCTIONS = [bucket_diff_top_k, get_entropy_feature]
    N_CLUSTERS = 3

    BATCH_SIZE = 16
    EPOCHS = 50
    LR = 5e-5

    GENERATE_DATA = False
    TRAIN_CLASSIFIER = False

    generate_data(functions=FUNCTIONS, bucket_indices=BUCKET_INDICES)

    # pipeline(functions=FUNCTIONS, bucket_indices=BUCKET_INDICES,
    #                  num_clusters=N_CLUSTERS, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR,
    #                  generate_data=GENERATE_DATA, train_classifier=TRAIN_CLASSIFIER)

    #print(f"final score: {score}")
