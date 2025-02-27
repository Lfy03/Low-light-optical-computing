import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt  # For displaying the image

# Device selection: GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Define convolution operation with the mask
class ConvWithMask(nn.Module):
    def __init__(self, mask):
        super(ConvWithMask, self).__init__()
        # Convert mask to tensor and adjust shape to match convolution kernel
        self.mask = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, H, W)

    def forward(self, x):
        # Convolve the input image x (shape: batch_size, 1, 28, 28) with the mask
        conv_output = F.conv2d(x, self.mask, padding=self.mask.shape[2] // 2)
        return conv_output

# 2. Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 3. Define input image size, mask size, and modulated size
input_size = (28, 28)
mask_size = (511, 511)
modulated_size = (256, 256)

# 4. Generate a mask (size: 511x511)
## mask = np.random.rand(*mask_size)  # Generate a random mask
r = 0.2
dp = 5.86e-3
nx = 511
ny = 511
Lx = dp * nx
Ly = dp * ny

x = np.linspace(-Lx/2, Lx/2 - dp, nx)
y = np.linspace(-Ly/2, Ly/2 - dp, ny)

xm, ym = np.meshgrid(x, y)

bi = np.pi / (r**2)
mask = 0.5 + 0.5 * np.cos(bi * (xm**2 + ym**2))

mask = torch.tensor(mask, dtype=torch.float32, device=device)  # Convert to tensor

# 5. Initialize convolution layer and move to the device (GPU/CPU)
conv_layer = ConvWithMask(mask).to(device)

# 6. List to store convolution results
preprocessed_images = []
preprocessed_labels = []

# 7. Iterate through the dataset
for images, labels in train_loader:
    images = images.to(device)  # Move input images to GPU
    upsampled_image = F.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False)
    # Perform convolution for each input image
    conv_result = conv_layer(upsampled_image)

    # Append the cropped result to the output list
    preprocessed_images.append(conv_result)
    preprocessed_labels.append(labels)
    #output_list.append(conv_result)

# 8. Convert the list of results into a multi-dimensional tensor
preprocessed_images = torch.cat(preprocessed_images, dim=0)
preprocessed_labels = torch.cat(preprocessed_labels, dim=0)
#output_tensor = torch.cat(output_list, dim=0)

# 9. Save the results as a .pt file
torch.save((preprocessed_images, preprocessed_labels), '/mnt/d/Program/Python_Algorithm/Low-light-optical-computing/data/temp/train_with_labels.pt')

print("Convolution results saved as '.pt' file.")

# 10. Display the first convolution result (showing the first image of the batch)
# Convert the first result to a numpy array for displaying
first_result = preprocessed_images[0, 0].cpu().detach().numpy()  # Get the first image, remove GPU tensor, and convert to numpy

# Plot the first convolution result
plt.imshow(first_result, cmap='gray')  # Use 'gray' colormap for grayscale image
plt.title('First Convolution Result')
plt.colorbar()  # Show color bar
plt.savefig('/mnt/d/Program/Python_Algorithm/Low-light-optical-computing/data/temp/convolution_result.png')
plt.close()