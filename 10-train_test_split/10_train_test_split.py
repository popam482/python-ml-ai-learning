import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = np.array([
    [5000, 4800], [4200, 4500], [6000, 5900], [3000, 3500], [5500, 5000],
    [4000, 4100], [4800, 4600], [5200, 5500], [3800, 3700], [6100, 6500]
])

y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#initialize logistic regression
classifier = LogisticRegression()

# train data only for training
classifier.fit(X_train, y_train)

prediction_test = classifier.predict(X_test)

accuracy_test = accuracy_score(y_test, prediction_test)
print(accuracy_test)

prediction_train = classifier.predict(X_train)

accuracy_train = accuracy_score(y_train, prediction_train)
print(accuracy_train)
