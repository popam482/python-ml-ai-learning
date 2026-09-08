import torch
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