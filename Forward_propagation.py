## Simulation

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# device choose
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data parameter
num_images = 7500     # image number train 5000 test 2500
batch_size = 64       # batch size
image_size = 28       # input image size
mask_size = 512       # mask size 
output_size = 256     # output size
padding_size = (mask_size - image_size) // 2  # 0 padding size

# 10000 input images, (num_images, 1, 28, 28)
images = torch.rand((num_images, 1, image_size, image_size), device=device)
data_dir = ""
np.load(data_dir)
torch.tensor()

# 1 mask, (1, 1, 512, 512)
mask = torch.rand((1, 1, mask_size, mask_size), device=device)

# 1. 0 padding 512x512 
padded_images = torch.zeros((num_images, 1, mask_size, mask_size), device=device)
padded_images[:, :, padding_size:padding_size + image_size, padding_size:padding_size + image_size] = images

# 2. nn.Conv2d
conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=mask_size, stride=1, padding=0, bias=False)
conv_layer.weight = nn.Parameter(mask)  # set mask as the kernal
conv_layer = conv_layer.to(device)
conv_layer.eval()  # 关闭梯度计算以提高推理速度

# 3. batch calculating
start_time = time.time()
outputs = []

with torch.no_grad():  # 关闭梯度计算
    for i in range(0, num_images, batch_size):
        batch = padded_images[i:i + batch_size]  # 获取 batch
        conv_result = conv_layer(batch)  # 卷积运算
        start_out = (conv_result.shape[-1] - output_size) // 2  # 计算裁剪起点
        cropped_result = conv_result[:, :, start_out:start_out + output_size, start_out:start_out + output_size]  # 中心裁剪
        outputs.append(cropped_result)  # 存入列表

outputs = torch.cat(outputs, dim=0)  # 拼接所有 batch 结果
end_time = time.time()

print(f"Processed {num_images} images in {end_time - start_time:.2f} seconds.")

# 4. save result
torch.save(outputs, "/mnt/d/Program/Python_Algorithm/Low-light-optical-computing/data/temp/conv_results.pt")
print("Results saved to 'conv_results.pt'")