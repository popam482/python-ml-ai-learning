import numpy as np
from sklearn.cluster import KMeans

# X = [Money_spent, Time_spent]
X_clients = np.array([
    [500, 10], [600, 15],
    [30, 120], [40, 100],
    [200, 60], [250, 50]
])

model = KMeans(n_clusters=3)

model.fit(X_clients)

predictions = model.predict(X_clients)

print(predictions)