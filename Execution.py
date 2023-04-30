from GetClusters.differenceMetrics import bucket_diff_top_k
from GetClusters.differenceMetrics import get_entropy_feature
from GetClusters.differenceMetrics import get_top_p_difference
from pipeline import pipeline

"""
If you are using e.g. the bucket transformation function, and therefore don't need the parameter TOP_P,
either leave it the way it is or set it to None
"""

# -------- hyperparameters -------- #

FUNCTION = get_top_p_difference  # bucket_diff_top_k  # get_entropy_feature
BUCKET_INDICES = [10, 35]  # if bucket_diff_top_k, choose where buckets should start and end,
# NB: 0 and len(distribution) are added automatically later!
TOP_P = 0.9  # probability for get_top_p_difference transformation function
N_CLUSTERS = 3  # number of clusters to use for k-means clustering, if None: no clustering or classifying!
BATCH_SIZE = 16
EPOCHS = 15
LR = 5e-5
GENERATE_DATA = True  # if True, generates probability distributions as training data, and features
GENERATE_SORTED_BY_BIG = False  # if True, generate probability distributions sorted by big model, for baseline
TRAIN_CLASSIFIER = True  # if True, trains a new classifier
N_TEST_SAMPLES = 500  # number of samples used for testing,
# each sample consists of a sequence of 64 distributions

# ----- end of hyperparameters ----- #

if __name__ == "__main__":

    final_score = pipeline(function=FUNCTION, n_clusters=N_CLUSTERS, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR,
                           generate_data=GENERATE_DATA, generate_sorted_by_big=GENERATE_SORTED_BY_BIG,
                           train_classifier=TRAIN_CLASSIFIER, bucket_indices=BUCKET_INDICES, top_p=TOP_P,
                           n_test_samples=N_TEST_SAMPLES)
    print(final_score)
