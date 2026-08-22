import numpy as np
from sklearn.ensemble import IsolationForest

X_sensors = np.array([
    [50, 20],
    [52, 21],
    [48, 19],
    [51, 20],
    [500, 80]
])

model = IsolationForest()
model.fit(X_sensors)

prediction = model.predict(X_sensors)

print(prediction)
