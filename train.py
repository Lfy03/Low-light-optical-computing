import torchvision
import torchvision.transforms as transforms

# 下载 MNIST 数据集
mnist_train = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())