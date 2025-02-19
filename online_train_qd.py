import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import os
from torch.utils.data import DataLoader, Dataset

# ===========================
#  1. Online Learning Dataset Loader
# ===========================

class OnlineTensorDataset(Dataset):
    """
    Custom dataset for online learning, loads .pt files batch-by-batch.
    Each sample consists of 32 stacked images (32, 256, 256).
    """
    def __init__(self, data_dir, labels, train=True, transform=None):
        self.data_dir = data_dir  # Path to .pt files
        self.labels = labels  # Corresponding labels
        self.transform = transform
        self.train = train  # If True, load first 5000 samples; else load last 2500
        self.file_list = sorted(os.listdir(data_dir))  # Sorted list of .pt files
        self.current_file_index = 0  # Track which .pt file is being processed
        self.current_data = None  # Data from the current .pt file
        self.load_next_file()  # Load first file

    def load_next_file(self):
        """Loads the next .pt file for processing."""
        if self.current_file_index >= len(self.file_list):
            return False  # No more files

        file_path = os.path.join(self.data_dir, self.file_list[self.current_file_index])
        self.current_data = torch.load(file_path)  # Shape: (256, 256, 7500)
        self.current_data = self.current_data.unsqueeze(0)  # Add batch dimension: (1, 256, 256, 7500)

        self.num_samples = 5000 if self.train else 2500
        self.start_index = 0 if self.train else 5000
        self.current_file_index += 1  # Move to the next file

        return True  # File loaded successfully

    def __len__(self):
        return self.num_samples * len(self.file_list)

    def __getitem__(self, idx):
        """Get one sample, switching to the next .pt file if needed."""
        if idx >= self.num_samples:
            if not self.load_next_file():
                raise StopIteration  # End of dataset

        idx += self.start_index  # Adjust for train/test split
        sample = self.current_data[:, :, :, idx]  # Shape: (1, 256, 256)
        sample = sample.squeeze(0)  # Remove batch dimension → (256, 256)

        label = self.labels[idx]  # Get label
        if self.transform:
            sample = self.transform(sample)

        return sample, label


# ===========================
#  2. Feature Extractor (CNN - ResNet)
# ===========================

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, output_dim=512):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)  # Reduce to 512D

    def forward(self, x):
        batch_size, h, w = x.shape  # (batch, 256, 256)
        x = x.unsqueeze(1)  # Add channel dimension → (batch, 1, 256, 256)
        x = self.feature_extractor(x)  # Shape: (batch, 512, 1, 1)
        x = torch.flatten(x, start_dim=1)  # Shape: (batch, 512)
        x = self.fc(x)  # Output feature vector
        return x


# ===========================
#  3. Denoising Autoencoder (DAE)
# ===========================

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, noise_factor=0.1):
        noisy_x = x + noise_factor * torch.randn_like(x)
        encoded = self.encoder(noisy_x)
        decoded = self.decoder(encoded)
        return decoded, encoded


# ===========================
#  4. Transformer Encoder
# ===========================

class TransformerEncoderModel(nn.Module):
    def __init__(self, feature_dim=256, num_heads=8, num_layers=2, dropout=0.1):
        super(TransformerEncoderModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(feature_dim, 32)  # 32-class classification

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)  # Global pooling
        x = self.classifier(x)
        return x


# ===========================
#  5. Full Model Pipeline
# ===========================

class FullModel(nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        self.resnet = ResNetFeatureExtractor(output_dim=512)
        self.dae = DenoisingAutoencoder(input_dim=512, hidden_dim=256)
        self.transformer = TransformerEncoderModel(feature_dim=256)

    def forward(self, x):
        features = self.resnet(x)
        denoised_features, encoded_features = self.dae(features)
        output = self.transformer(encoded_features)
        return output


# ===========================
#  6. Online Training Function
# ===========================

def train_online(model, data_dir, labels, device, epochs=10, lr=1e-4, save_path="online_model.pth"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        dataset = OnlineTensorDataset(data_dir, labels, train=True)  # Load dataset for each epoch
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

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

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")
        torch.save(model.state_dict(), save_path)


# ===========================
#  7. Execute Online Training
# ===========================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels_all = torch.load("path_to_labels.pt")  # Shape: (7500 × N)
    labels_train = labels_all[:5000, :]  # get 5000 as train set

    model = FullModel()
    train_online(model, "data/.", labels_train, device, epochs=10, save_path="online_model.pth")