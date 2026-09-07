import torch
import torch.nn as nn
import torch.optim as optim

# y = 3x +2

# 2d tensor - data
X = torch.linspace(-5, 5, 100).unsqueeze(1)
y = 3 * X + 2


# model define
class LinearRegressionModel(nn.Module):  # inherits torch.nn.Module
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x)


model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()  # reset gradients
    y_pred = model(X)  # forward ppass - calculates predictions
    loss = criterion(y_pred, y)  # calculate error
    loss.backward()  # backward pass
    optimizer.step()  # optimisation

print(f"Final loss: {loss.item():.4f}")
print(
    f"Learned weight W: {model.linear.weight.item():.2f} (Expected: ~3.00)"
)
print(f"Learned bias b: {model.linear.bias.item():.2f} (Expected: ~2.00)")

# y1 = x^2 - 4

X1 = torch.linspace(-3, 3, 200).unsqueeze(1)
y = X1 ** 2 - 4

# model define
class NonlinearRegressionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(
            in_features=1, out_features=10
        )
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(
            in_features=10, out_features=1
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

model = NonlinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(500):
    optimizer.zero_grad()
    y_pred = model(X1)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

print(f"Final loss: {loss.item():.4f}")