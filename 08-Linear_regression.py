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

X1 = np.array([
    [50, 2, 10],
    [30, 1, 40],
    [80, 3, 30],
    [65, 2, 35]
])

y1 = np.array([70000, 450000, 110000, 65000])

model1 = LinearRegression()
model1.fit(X1, y1)

new_apartment = np.array([[70, 4, 45]])

y_new_apartment = model1.predict(new_apartment)

print(y_new_apartment)