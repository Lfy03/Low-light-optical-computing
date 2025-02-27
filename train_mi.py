import numpy as np
import random
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc      = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
def train_batch(model, train_loader, val_loader, device, epochs=5, lr=0.001):
    """
    Batch Training Process
    Args:
        model: CNN model
        train_loader: DataLoader for training
        device: 'cuda' or 'cpu'
        epochs: Number of training epochs
        lr: Learning rate
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    scaler = GradScaler()

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    total_start_time = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_start_time = time.time()

        model.train()
        total_loss, correct, total = 0, 0, 0
        
        batch_bar = tqdm(train_loader, desc="Training", leave=False)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
            with autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
            batch_bar.update()

        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"Train_Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"Val_Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

        scheduler.step(val_acc)
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds.\n")

    total_duration = time.time() - total_start_time
    print(f"Total training time: {total_duration:.2f} seconds.")
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig("training_metrics.png")

    return model


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the trained model on test data.
    """
    model.to(device)
    model.eval()

    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss/len(test_loader), 100*correct/total

set_seed(42)

# 加载数据集 预处理后的数据的尺寸为 256x256
preprocessed_train, train_labels = torch.load("/mnt/d/Program/Python_Algorithm/Low-light-optical-computing/data/MNIST/train/train_with_labels.pt")
preprocessed_test, test_labels = torch.load("/mnt/d/Program/Python_Algorithm/Low-light-optical-computing/data/MNIST/test/test_with_labels.pt")

preprocessed_train = preprocessed_train.cpu()
train_labels = train_labels.cpu()
preprocessed_test = preprocessed_test.cpu()
test_labels = test_labels.cpu()

train_dataset = TensorDataset(preprocessed_train, train_labels)
train_subset, val_subset = torch.utils.data.random_split(train_dataset, [50000, 10000])
test_dataset = TensorDataset(preprocessed_test, test_labels)

# 设定批量大小
batch_size = 64  # 可以调整 batch_size 大小
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

# 训练
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True  # 启用cudnn自动优化器
torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32加速
model = SimpleCNN(num_classes=10)
model = train_batch(model, train_loader, val_loader, device, epochs=100, lr=0.001)
torch.save(model.state_dict(), "/mnt/d/Program/Python_Algorithm/Low-light-optical-computing/model/temp/batch_model.pth")
print("Model saved successfully!")

# 评估
criterion = nn.CrossEntropyLoss()
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Final Test Accuracy: {test_acc:.2f}%")