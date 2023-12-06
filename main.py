import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

# Read the data
with open("./data/input01.txt", "r", encoding='utf-8') as input_text:
    text = input_text.read()
print("the length of this text is:", len(text))

chars = sorted(list(set(text)))
for i in range(chars.index(".")):
    chars.append(chars.pop(0))

char_size = len(chars)
print(''.join(chars))
print(f'there are {char_size} characters as shown above')

### Constant ###
BATCH_SIZE = 8
CHAR_NUM = len(chars)
TEXT_SIZE = len(text)
CHANNEL_SIZE = 16

# create encoder and decoder for ".abcdefghijklmnopqrstuvwxyz \n"
ston = { ch:i for i,ch in enumerate(chars) }
ntos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [float(ston[c])/(CHAR_NUM - 1) for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([ntos[round(i * (CHAR_NUM - 1))] for i in l]) # decoder: take a list of integers, output a string


# target text with shape (BATCH_SIZE,)
target_ntext = torch.tensor(encode(text))[..., None]
target_ntext = torch.concat((target_ntext, torch.ones_like(target_ntext)), dim=-1)
orig = decode(target_ntext[:,0].squeeze().tolist())
print(orig)

# Initialization: the first character is "."
# shape (BATCH_SIZE, TEXT_SIZE, CHANNEL_SIZE) 

init_ntext = torch.zeros((BATCH_SIZE, TEXT_SIZE, CHANNEL_SIZE))
init_ntext[:, 0, 1:] = 1
print(init_ntext.shape)

#### Model yet to be defined
#class CANN(nn.Module):
#    """"""
#    def __init__(self, n_channels, cell_survival_rate):
#        super().__init__()
#        self.n_channels = n_channels
#        self.cell_survival_rate = cell_survival_rate
#
#        self.seq = nn.Sequential(
#            nn.Conv2d(n_channels * 3, 128, kernel_size=1),
#            nn.ReLU(),
#            nn.Conv2d(128, n_channels, kernel_size=1, bias=False),
#        )
#
#        # initialize weights to zero to prevent random noise
#        with torch.no_grad():
#            self.seq[2].weight.zero_()
#
#    def perceived_vector(self, X):
#        sobel_filter_ = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#        scalar = 8.0
#
#        sobel_filter_x = sobel_filter_ / scalar
#        sobel_filter_y = sobel_filter_.t() / scalar
#        identity_filter = torch.tensor(
#                [
#                    [0, 0, 0],
#                    [0, 1, 0],
#                    [0, 0, 0],
#                ],
#                dtype=torch.float32,
#        )
#        filters = torch.stack([identity_filter, sobel_filter_x, sobel_filter_y])  # (3, 3, 3)
#        filters = filters.repeat((16, 1, 1))  # (3 * n_channels, 3, 3)
#        stacked_filters = filters[:, None, ...].to(device) # (3 * n_channels, 1, 3, 3)
#
#        perceived = F.conv2d(X, stacked_filters, padding=1, groups=self.n_channels)
#
#        return perceived
#
#    def stochastic_update(self, X):
#        mask = (torch.rand(X[:, :1, :, :].shape) <= self.cell_survival_rate).to(device, torch.float32)
#        return X * mask
#
#    def live_cell_mask(self, X, alpha_threshold=0.1):
#        val = F.max_pool2d(X[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > alpha_threshold
#        return (val)
#
#    def forward(self, X):
#        pre_mask = self.live_cell_mask(X)
#        y = self.perceived_vector(X)
#        dx = self.seq(y)
#        dx = self.stochastic_update(dx)
#
#        X = X + dx
#
#        post_mask = self.live_cell_mask(X)
#        live_mask = (pre_mask & post_mask).to(torch.float32)
#
#        return X * live_mask
