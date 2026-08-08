import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print(f"Data dimensions: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

prediction = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, prediction)
print(f"Accuracy: {accuracy*100:.2f}")
