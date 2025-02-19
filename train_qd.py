import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os

# ===========================
#  1. Dataset Class for Loading .pt Files
# ===========================

class TensorDataset(Dataset):
    """
    Custom dataset for loading .pt files containing 256x256x7500 tensors.
    Each sample consists of 32 stacked images (32, 256, 256).
    """
    def __init__(self, data_dir, labels, train=True, transform=None):
        self.data_dir = data_dir  # Path to .pt files
        self.labels = labels  # Corresponding labels for classification
        self.transform = transform
        self.train = train  # If True, use first 5000 samples; else use last 2500

        # Load all .pt files into memory
        self.file_list = sorted(os.listdir(data_dir))  # 32 .pt files
        self.data = [torch.load(os.path.join(data_dir, f)) for f in self.file_list]
        self.data = torch.stack(self.data, dim=0)  # Shape: (32, 256, 256, 7500)

        # Select training or test split
        self.num_samples = 5000 if train else 2500
        self.start_index = 0 if train else 5000

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx += self.start_index  # Offset based on train/test split

        # Extract 32 images for the current sample, shape: (32, 256, 256)
        tensors = self.data[:, :, :, idx]

        # Get corresponding label
        y = self.labels[idx]

        # Apply transformation (if any)
        if self.transform:
            tensors = self.transform(tensors)

        return tensors, y


# ===========================
#  2. Feature Extractor (CNN - ResNet)
# ===========================

class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-based CNN feature extractor.
    Extracts 512-dimensional feature vectors from 256x256 images.
    """
    def __init__(self, output_dim=512):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)  # Pre-trained ResNet18
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)  # Reduce to 512D

    def forward(self, x):
        batch_size, num_images, h, w = x.shape  # (batch, 32, 256, 256)
        x = x.view(batch_size * num_images, 1, h, w)  # Reshape for CNN processing
        x = self.feature_extractor(x)  # Output shape: (batch*32, 512, 1, 1)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch*32, 512)
        x = self.fc(x)  # Map to output_dim (512)
        return x.view(batch_size, num_images, -1)  # Reshape to (batch, 32, 512)


# ===========================
#  3. Denoising Autoencoder (DAE)
# ===========================

class DenoisingAutoencoder(nn.Module):
    """
    Denoising autoencoder (DAE) for feature enhancement.
    """
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
        noisy_x = x + noise_factor * torch.randn_like(x)  # Add Gaussian noise
        encoded = self.encoder(noisy_x)  # Compress to 256D
        decoded = self.decoder(encoded)  # Reconstruct to 512D
        return decoded, encoded


# ===========================
#  4. Transformer Encoder
# ===========================

class TransformerEncoderModel(nn.Module):
    """
    Transformer encoder for sequence modeling of extracted features.
    """
    def __init__(self, feature_dim=256, num_heads=8, num_layers=2, dropout=0.1):
        super(TransformerEncoderModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(feature_dim, 32)  # 32-class classification

    def forward(self, x):
        x = self.transformer_encoder(x)  # Transformer processing
        x = torch.mean(x, dim=1)  # Global pooling
        x = self.classifier(x)  # Classification
        return x


# ===========================
#  5. Full Model Pipeline
# ===========================

class FullModel(nn.Module):
    """
    Complete pipeline: ResNet → DAE → Transformer → Classification.
    """
    def __init__(self):
        super(FullModel, self).__init__()
        self.resnet = ResNetFeatureExtractor(output_dim=512)
        self.dae = DenoisingAutoencoder(input_dim=512, hidden_dim=256)
        self.transformer = TransformerEncoderModel(feature_dim=256)

    def forward(self, x):
        features = self.resnet(x)  # CNN feature extraction
        denoised_features, encoded_features = self.dae(features)  # Denoising & encoding
        output = self.transformer(encoded_features)  # Classification
        return output


# ===========================
#  6. Training Function
# ===========================

def train_model(model, train_loader, device, epochs=10, lr=1e-4, save_path="model.pth"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch in train_loader:
            images, labels = batch
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
#  7. Execute Training
# ===========================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels_all = torch.load("path_to_labels.pt")  # Shape: (7500 × N)
    labels_train = labels_all[:5000, :]  # get 5000 as train set

    dataset = TensorDataset("data/.", labels_train, train=True)  # Training dataset
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = FullModel()
    train_model(model, train_loader, device, epochs=10, save_path="model.pth")