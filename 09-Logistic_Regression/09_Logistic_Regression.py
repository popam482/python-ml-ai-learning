from sklearn.linear_model import LogisticRegression
import numpy as np

X_electrical_grid = np.array([
    [1200, 800],
    [200, 100],
    [1500, 50],
    [100, 0]
])

# 0-surplus 1-deficit
y_state = np.array([0, 1, 0, 1])

new_hour = np.array([[800, 400]])

classifier = LogisticRegression()
classifier.fit(X_electrical_grid, y_state)

prediction = classifier.predict(new_hour)
print(prediction)