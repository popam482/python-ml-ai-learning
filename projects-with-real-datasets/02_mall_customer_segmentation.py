import pandas as pd
from sklearn.preprocessing import StandardScaler


def read_data():
    df = pd.read_csv("../datasets/Mall_Customers.csv")
    return df

def data_scaling(df):
    features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)
    return scaled_data

def main():
    df = read_data()
    scaled_data = data_scaling(df)
    print(scaled_data)

if __name__ == "__main__":
    main()
