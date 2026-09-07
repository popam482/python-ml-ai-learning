import torch
import torch.nn as nn
import torch.optim as optim

# y = 3x +2

# 2d tensor - data
X = torch.linspace(-5, 5, 100).unsqueeze(1)
y = 3 * X + 2


# model define
class LinearRegressionModel(nn.Module): # inherits torch.nn.Module
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad() # reset gradients
    y_pred = model(X) # forward ppass - calculates predictions
    loss = criterion(y_pred, y) # calculate error
    loss.backward() # backward pass
    optimizer.step() # optimisation

print(f"Final loss: {loss.item():.4f}")
print(
    f"Learned weight W: {model.linear.weight.item():.2f} (Expected: ~3.00)"
)
print(f"Learned bias b: {model.linear.bias.item():.2f} (Expected: ~2.00)")
