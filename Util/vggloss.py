import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
from torchvision import models
import os, cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGGLoss(nn.Module):
    def __init__(self, device=device, n_layers=5):
        super().__init__()

        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)
        vgg = models.vgg19(pretrained=True).features

        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer].to(device))
            self.layers.append(layers.to(device))
            prev_layer = next_layer

        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().to(device)

    def forward(self, source, target):
        loss = 0
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight * self.criterion(source, target)

        return loss

# if __name__ == "__main__":
#     print(device)
#     fea_save_path = "./feature_save/"
#     if not os.path.exists(fea_save_path):
#         os.mkdir(fea_save_path)
#     img1 = np.array(cv2.imread("1.png")) / 255.0
#     img2 = np.array(cv2.imread("2.png")) / 255.0
#     img1 = img1.transpose((2, 0, 1))
#     img2 = img2.transpose((2, 0, 1))
#     print(img1.shape, img2.shape)
#     print(type(img1))
#     img1_torch = torch.unsqueeze(torch.from_numpy(img1), 0)
#     img2_torch = torch.unsqueeze(torch.from_numpy(img2), 0)
#     img1_torch = torch.as_tensor(img1_torch, dtype=torch.float32)
#     img2_torch = torch.as_tensor(img2_torch, dtype=torch.float32)
#     total_perceptual_loss = VGGLoss('cpu')
#     print(img1_torch.shape)
#     print(type(img1_torch))
#     loss1 = total_perceptual_loss(img1_torch,img2_torch)
#     print(loss1)

if __name__ == '__main__':
    x=torch.randn(1,3,256,256)
    y=torch.randn(1,3,256,256)

    loss=VGGLoss()
    z=loss(x.to('cuda'),y.to('cuda'))
    z_value=z.item()
    print(z_value)