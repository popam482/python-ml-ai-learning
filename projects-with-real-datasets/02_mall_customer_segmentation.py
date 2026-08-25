import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import (StandardScaler)
from sklearn.cluster import KMeans


def read_data():
    df = pd.read_csv("../datasets/Mall_Customers.csv")
    return df


def data_scaling(df):
    features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)
    return scaled_data


def elbow_plot(inertia):
    plt.plot(range(1, 10), inertia, marker='o')
    plt.title('Elbow plot to determine K')
    plt.xlabel('Number of clusters - K')
    plt.ylabel('Inertia')
    plt.show()


def find_optimal_number_of_clusters(scaled_data):
    inertia = []
    for k in range(1, 10):
        model_kmeans = KMeans(n_clusters=k, random_state=42)

        model_kmeans.fit(scaled_data)

        inertia.append(model_kmeans.inertia_)

    elbow_plot(inertia)


def k_means_training(scaled_data, k_value):
    km = KMeans(n_clusters=k_value, random_state=42)
    return km.fit_predict(scaled_data)


def add_tag_to_df(df, scaled_data, k_value):
    cluster_ids = k_means_training(scaled_data, k_value)
    df['cluster_ID'] = cluster_ids
    return cluster_ids


def pca_plot(pca_data, cluster_ids):
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        pca_data[:, 0],
        pca_data[:, 1],
        c=cluster_ids,
        cmap='viridis',
        edgecolors='black',
        linewidths=1
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster ID')

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('K-means clustering with PCA')

    plt.show()


def apply_pca(scaled_data, cluster_ids):
    model = PCA(n_components=2)

    pca_data = model.fit_transform(scaled_data)

    print(pca_data)

    pca_plot(pca_data, cluster_ids)

def cluster_profiling(df):
    profile = df.groupby('cluster_ID')[
        ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    ].mean()

    print(profile)

    return profile

def main():
    df = read_data()
    scaled_data = data_scaling(df)
    print(scaled_data)
    find_optimal_number_of_clusters(scaled_data)
    # after observing the plot: k=4
    k_value = 4
    cluster_ids = add_tag_to_df(df, scaled_data, k_value)
    apply_pca(scaled_data, cluster_ids)
    cluster_profiling(df)

if __name__ == "__main__":
    main()
