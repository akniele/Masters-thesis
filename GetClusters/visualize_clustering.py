import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

# Generate random data with 3 features
np.random.seed(0)
X = np.random.randn(100, 3)

kmeans = KMeans(
            init="random",
            n_clusters=3,
            n_init=10,
            max_iter=300,
            random_state=42
            )
kmeans.fit(X)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Plot the data and cluster centers in a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels)
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='x', s=200, linewidths=3, color='r')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')

plt.savefig("try.png")
