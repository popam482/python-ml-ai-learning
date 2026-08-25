import pandas as pd
import matplotlib.pyplot as plt
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



def main():
    df = read_data()
    scaled_data = data_scaling(df)
    print(scaled_data)
    find_optimal_number_of_clusters(scaled_data)
    # after observing the plot: 4
    k_value = 4
    prediction = k_means_training(scaled_data, k_value)
    print(prediction)

if __name__ == "__main__":
    main()
