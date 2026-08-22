import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


X_7_days = np.array([
    [1500, 900], [1600, 850], [1400, 950], [1550, 800],
    [900, 500],
    [200, 100], [100, 50]
])


y_real = np.array([0, 0, 0, 0, 0, 1, 1])


classifier = LogisticRegression()
classifier.fit(X_7_days, y_real)


prob = classifier.predict_proba(X_7_days)

prob_deficit = prob[:, 1]
print(f"Deficit chances: {prob_deficit}")

# anything above 20% is true
custom_prediction = (prob_deficit >= 0.2).astype(int)

new_matrix = confusion_matrix(y_real, custom_prediction)
print(f"New matrix (20%):\n {new_matrix}")