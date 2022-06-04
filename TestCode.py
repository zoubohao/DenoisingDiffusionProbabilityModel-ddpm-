import torch
import torch.nn as nn

a = torch.randn(size=[8, 3, 4, 4])
m = nn.ConvTranspose2d(3, 3, 5, 2, 2, 1)
b = m(a)
print(b.shape)