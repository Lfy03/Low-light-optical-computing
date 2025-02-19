import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class OnlineDataset(Dataset):
    def __init__(self, dataset, start_idx, end_idx, transform=None):
        self.dataset = dataset
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.transform = transform

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        idx += self.start_idx
        image, label = self.dataset[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
def train_online(model, dataset, device, batch_size=64, epochs=5, lr=0.001):
    """
    Online Learning Training Process
    Args:
        model: CNN model
        dataset: Entire dataset (CIFAR-10 / FashionMNIST)
        device: 'cuda' or 'cpu'
        batch_size: Size of each mini-batch
        epochs: Number of training epochs
        lr: Learning rate
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # 在线训练，每个 epoch 加载部分数据
        train_data = OnlineDataset(dataset, start_idx=0, end_idx=5000, transform=train_transform)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"Loss: {total_loss / len(train_loader):.4f}, Accuracy: {acc:.2f}%")

    torch.save(model.state_dict(), "online_model.pth")
    print("Model saved successfully!")

def evaluate(model, dataset, device, batch_size=64):
    """
    Evaluate the trained model on test data.
    """
    model.to(device)
    model.eval()

    test_data = OnlineDataset(dataset, start_idx=5000, end_idx=7500, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")

# 数据变换
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),  
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 选择数据集: FashionMNIST / CIFAR-10
dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=None)

# 训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=10)
train_online(model, dataset, device, batch_size=64, epochs=5, lr=0.001)

# 评估
evaluate(model, dataset, device)
