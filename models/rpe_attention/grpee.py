# %%
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, GELU
import torch.nn.functional as F

class grpee(nn.Module):
    def __init__(self, in_channel, out_channel, heads):
        super(grpee, self).__init__()
        self.pos_encoder = Sequential(Linear(in_channel, out_channel),
                                      GELU(),
                                      Linear(out_channel, heads*out_channel))
    
    def forward(self, x, position):
        # The shape of x is (L, H, C)
        # The shape of position is (L, in_channel)
        L, H, C = x.shape
        rpe = self.pos_encoder(position)
        rpe = rpe.view(L, H, C)
        return (rpe * x).sum(dim=2)