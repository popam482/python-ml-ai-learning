import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#load the data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

#split the data - 80% train 20% test
# random_state set to 42 in order to have the same cut for every script run
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardised data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# model initialization and training
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# predictions
predictions = model.predict(X_test_scaled)

#evaluation
accuracy = accuracy_score(y_test, predictions)
print(f"The model accuracy is {accuracy * 100:.2f}%")