from sklearn.linear_model import LinearRegression
import numpy as np

# X input data
X = np.array([[50], [60], [80], [100]])

# y what we want to predict
y = np.array([[50000], [65000], [75000], [120000]])

# model initialization
model = LinearRegression()

# train the model
model.fit(X, y)

# prediction for a 70 square meters house
estimated_price = model.predict([[70]])

print(estimated_price)
