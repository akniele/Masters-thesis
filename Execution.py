from GetClusters.differenceMetrics import bucket_diff_top_k
from GetClusters.differenceMetrics import get_entropy_feature
from GetClusters.differenceMetrics import get_top_p_difference
from pipeline import run_transparent_pipeline
from pipeline import run_baseline
from pipeline import make_directories
from argparse import ArgumentParser, ArgumentTypeError

parser = ArgumentParser()

parser.add_argument("pipeline", type=str, choices=["transparent", "baseline"],
                    default="transparent",
                    help="specify whether to run the baseline model or the transparent pipeline")

parser.add_argument("--data", required=False, type=bool, default=False,
                    help="generate new training data and features")
parser.add_argument("--data2", required=False, type=bool, default=False,
                    help="generate new training data sorted by probs of big model")
parser.add_argument("--n_test", required=False, type=int, default=500,
                    help="the number of samples (sequences of 64 distributions) to test the transformations on")
parser.add_argument("--lr", required=False, type=float, default=5e-5,
                    help="the learning rate, either for the baseline model or for the classifier model")
parser.add_argument("--epochs", required=False, type=int, default=25,
                    help="the number of epochs, either for the baseline model or for the classifier model")
parser.add_argument("--batch", required=False, type=int, default=16,
                    help="the batch size, either for the baseline model or for the classifier model")
parser.add_argument("--random", required=False, type=bool, default=False,
                    help="if set to True, use randomly generated labels instead of labels from classifier")
parser.add_argument("--train_classifier", required=False, type=bool, default=False,
                    help="if set to True, train classifier model")
parser.add_argument("--n_clusters", required=False, type=int, default=3,
                    help="the number of clusters to use for clustering the feature differences,"
                    "should be either an int > 0, or 0 if you do not want to use clustering")
parser.add_argument("--top_p", required=False, type=float, default=0.47,
                    help="the probability mass p used for the top-p transformation function")
parser.add_argument("--indices", required=False, default=[1, 2], type=int, nargs="+",
                    help="the bucket indices for the bucket transformation function, pass as a string with"
                    "a space in between each number")
parser.add_argument("--function", required=False, choices=[1, 2, 3], type=int, default=1,
                    help="the transformation function to use, either 1 for 'get_top_p_difference', or"
                    "2 for 'get_entropy_feature', or 3 for 'bucket_diff_top_k'")

args = parser.parse_args()

generate_data = args.data  # if True, generates probability distributions as training data, and features
generate_sorted_by_big = args.data2  # if True, generate probability distributions sorted by big model, for baseline
n_test_samples = args.n_test   # number of samples used for testing

if args.function == 1:
    function = get_top_p_difference
elif args.function == 2:
    function = get_entropy_feature
elif args.function == 3:
    function = bucket_diff_top_k
else:
    raise ArgumentTypeError('The command line argument here needs to be 1, 2 or 3.')

bucket_indices = args.indices  # if bucket_diff_top_k, choose where buckets should start and end,
                               # NB: 0 and len(distribution) are added automatically later!
top_p = args.top_p  # probability for get_top_p_difference transformation function

if args.n_clusters == 0:
    n_clusters = None  # no clustering or classifying!
else:
    n_clusters = args.n_clusters  # number of clusters to use for k-means clustering

batch_size = args.batch  # 16 for classifier model, 8 for baseline model
epochs = args.epochs
lr = args.lr
train_classifier = args.train_classifier  # if True, trains a new classifier
random_labels = args.random


if __name__ == "__main__":

    make_directories()

    if args.pipeline == "baseline":
        run_baseline(n_test_samples=n_test_samples, batch_size=batch_size, epochs=epochs,
                     lr=lr, generate_data=generate_data, generate_sorted_by_big_data=generate_sorted_by_big)
    elif args.pipeline == "transparent":
        run_transparent_pipeline(function=function, n_clusters=n_clusters, batch_size=batch_size, epochs=epochs, lr=lr,
                                 generate_data=generate_data, generate_sorted_by_big=generate_sorted_by_big,
                                 train_classifier=train_classifier, bucket_indices=bucket_indices, top_p=top_p,
                                 n_test_samples=n_test_samples, random_labels=random_labels)
