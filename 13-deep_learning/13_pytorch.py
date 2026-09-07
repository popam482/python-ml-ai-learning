import torch

# define tensor and activate gradients
x = torch.tensor(2.0, requires_grad=True)

# forward pass
y = x ** 2 + 3 * x + 5

# backward pass - calculate gradients
y.backward()

print(f"Function value y(2): {y.item()}")
print(f"dy/dx derivative for x=2: {x.grad.item()}")

x1 = torch.tensor(3.0, requires_grad=True)
y1 = torch.tensor(2.0, requires_grad=True)

z = 2 * x1 ** 3 - 4 * y1 ** 2 + 7

z.backward()

print(f"dy/dx derivative for x=3: {x1.grad.item()}")
print(f"dx/dy derivative for y=2: {y1.grad.item()}")