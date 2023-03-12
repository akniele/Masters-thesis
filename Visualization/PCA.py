"""## Principal Component Analysis"""
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.decomposition import PCA


def pca_percentage2explain(scaled_features, percentage2explain=0.9):
    pca = PCA(percentage2explain)
    scaled_transformed_features = pca.fit_transform(scaled_features)
    return scaled_transformed_features, pca.n_components_, pca.explained_variance_ratio_


def pca_given_num_components(scaled_features, n_components=3):
    pca = PCA(n_components)
    scaled_transformed_features = pca.fit_transform(scaled_features)
    return scaled_transformed_features, pca.n_components_, pca.explained_variance_ratio_


def pca_plot(components, n_components, explained_variance, twoD=True):
    if n_components == 3:
        if twoD:
            sns.scatterplot(x=components[:, 0], y=components[:, 1],
                            hue=components[:, 2])
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.show()
        else:
            total_var = explained_variance.sum() * 100
            # color=components[:, 3]
            fig = px.scatter_3d(
                components, x=components[:, 0], y=components[:, 1], z=components[:, 2],
                title=f'Total Explained Variance: {total_var:.2f}%',
                labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
            )
            fig.show()
    elif n_components == 4:
        total_var = explained_variance.sum() * 100
        fig = px.scatter_3d(
            components, x=components[:, 0], y=components[:, 1], z=components[:, 2], color=components[:, 3],
            title=f'Total Explained Variance: {total_var:.2f}%',
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )
        fig.show()
    elif n_components == 2:
        sns.scatterplot(x=components[:, 0], y=components[:, 1])
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()
    else:
        print("Number of components not covered by this function (use either 2, 3 or 4).")
