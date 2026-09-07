import torch

#define tensor and activate gradients
x = torch.tensor(2.0, requires_grad=True)

# forward pass
y = x ** 2 + 3 * x + 5

# backward pass - calculate gradients
y.backward()

print(f"Function value y(2): {y.item()}")
print(f"dy/dx derivative for x=2: {y.grad.item()}")