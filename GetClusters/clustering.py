"""# K-means clustering

## get cluster labels
"""
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
import pandas as pd
import plotly.express as px
from collections import defaultdict


def k_means_clustering(feature_vector, n_clusters):
    print("  => GET CLUSTER LABELS")
    kmeans = KMeans(
            init="random",
            n_clusters=n_clusters,
            n_init=10,
            max_iter=300,
            random_state=42
            )
    kmeans.fit(feature_vector)

    labels = kmeans.labels_
    return labels


"""## Which k is best?"""


def find_optimal_n(feature_vector):
    print("  => FINDING OPTIMAL NUMBER OF CLUSTERS")
    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
        }
    # find elbow
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(feature_vector)
        sse.append(kmeans.inertia_)
    kl = KneeLocator(
        range(1, 11), sse, curve="convex", direction="decreasing"
        )
    elbow = kl.elbow

    return elbow


"""Check the distribution of cluster labels"""


def label_distribution(n_clusters, labels):
    print("  => GET LABEL DISTRIBUTION")
    n_occurrences = []
    for i in range(n_clusters):
        count = np.count_nonzero(labels == i)
        n_occurrences.append(count)
    return n_occurrences


def visualize_label_distribution(labels):
    label_dict = dict()
    label_dict['cluster_label'] = []
    label_dict['cluster_label'].extend(labels)
    label_df = pd.DataFrame.from_dict(label_dict)

    fig = px.histogram(label_df, x="cluster_label")
    fig.show()


"""## Visualize the probabilities for each cluster separately"""


CLUSTER_TO_VISUALIZE = 1  # first check how many clusters were used for clustering algorithm
SHEET_TO_VISUALIZE = 0  # there are 32 sheets


def visualize_cluster_probs(labels, probs1, probs0):
    visualization = defaultdict(list)
    for i, sheets in enumerate(probs1[:1]):
        for j, samples in enumerate(sheets):
            if labels[j] == CLUSTER_TO_VISUALIZE:  # if distribution pertains to cluster 1
                sorted_probs = sorted(probs1[i+SHEET_TO_VISUALIZE][j], reverse=True)
                sorted_probs = sorted_probs[:30]
                sorted_small = sorted(probs0[i+SHEET_TO_VISUALIZE][j], reverse=True)
                sorted_small = sorted_small[:30]
                for k in range(30):
                    visualization["model"].append("big")
                    visualization["probs"].append(sorted_probs[k])
                    visualization["timestep"].append(j)
                    visualization["token_id"].append(k)
                for l in range(30):
                    visualization["model"].append("small")
                    visualization["probs"].append(sorted_small[l])
                    visualization["timestep"].append(j)
                    visualization["token_id"].append(l)
    full_data = pd.DataFrame.from_dict(visualization)

    fig = px.line(full_data, x="token_id", y="probs", animation_frame="timestep", color="model",
                  range_x=[0, 29], range_y=[1e-7,1], log_y=True)
    fig["layout"].pop("updatemenus")  # optional, drop animation buttons
    fig.show()
