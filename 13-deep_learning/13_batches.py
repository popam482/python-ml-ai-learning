import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class CustomEnergyDataset(Dataset):
    def __init__(self, X_tensor, y_tensor):
        self.X_tensor = X_tensor
        self.y_tensor = y_tensor

    def __len__(self):
        return len(self.X_tensor)

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.y_tensor[idx]

X = torch.randn(1000, 5)
y = torch.randn(1000, 1)

dataset = CustomEnergyDataset(X, y)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

for batch_X, batch_y in dataloader:
    print("batch_X shape: ", batch_X.shape)
    print("batch_y shape: ", batch_y.shape)
    break


X_fictive = torch.randn(1000, 5)
y_fictive = torch.randn(1000, 1)

class CustomDataset(Dataset):
    def __init__(self, X_tensor, y_tensor):
        self.X_tensor = X_tensor
        self.y_tensor = y_tensor
    def __len__(self):
        return len(self.X_tensor)
    def __getitem__(self, idx):
        return self.X_tensor[idx], self.y_tensor[idx]

dataset = CustomDataset(X_fictive, y_fictive)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 1)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))

model = SimpleMLP()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()

for epoch in range(50):
    loss_current = 0.0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(model.forward(batch_X), batch_y)
        loss.backward()
        optimizer.step()
        loss_current += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/50, Loss: {loss_current:.4f}")

model.eval()
with torch.no_grad():
    test_X = X_fictive[:5]

    test_predictions = model(test_X)

    print("Model test")
    print("\n First 5 predictions: ")
    print(test_predictions)
