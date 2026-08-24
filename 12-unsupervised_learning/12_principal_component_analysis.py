import numpy as np
from sklearn.decomposition import PCA

X_redundant = np.array([
    [220, 220000, 10, 10000],
    [230, 230000, 12, 12000],
    [210, 210000, 11, 11000],
    [225, 225000, 15, 15000]
])

model = PCA(n_components=2) # 2 columns

X_compressed = model.fit_transform(X_redundant)

print(X_compressed)