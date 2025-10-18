import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, in_channel_high, in_channel_low, out_channel):
        super(AttentionBlock, self).__init__()

        self.in_channel_high = in_channel_high
        self.in_channel_low = in_channel_low
        self.out_channel = out_channel

        self.weight_low = nn.Sequential(
            nn.Conv2d(in_channel_high, out_channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channel)
        )
        self.weight_high = nn.Sequential(
            nn.Conv2d(in_channel_low, out_channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channel)
        )
        self.compute_attn_weight = nn.Sequential(
            nn.Conv2d(out_channel, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, low, high):
        low = self.weight_low(low)
        high = self.weight_high(high)
        fusion_weight = self.relu(low + high)
        attn_weight = self.compute_attn_weight(fusion_weight)

        if self.out_channel == self.in_channel_low:
            return high * attn_weight
        elif self.out_channel == self.in_channel_high:
            return low * attn_weight


class Decoder(nn.Module):
    def __init__(self, in_channel_high, in_channel_low, out_channel):
        super().__init__()

        in_channel_all = in_channel_low + in_channel_high

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel_all, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
        )
    def forward(self, low, high):
        low = F.interpolate(low, scale_factor=2, mode='bilinear')

        fusion = torch.cat([low, high], dim=1)

        out = self.conv(fusion)
        return out


class InteractionDecoder(nn.Module):
    def __init__(self, in_channel_high, in_channel_low, out_channel):
        super().__init__()

        in_channel_all = in_channel_low + in_channel_high

        # self.attn_block_diff = AttentionBlock(in_channel_high, in_channel_low, in_channel_high)
        # self.attn_block_cat = AttentionBlock(in_channel_high, in_channel_low, in_channel_low)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel_all, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel)
        )

        self.identity = nn.Sequential(
            nn.Conv2d(in_channel_all, out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel)
        )
        self.relu = nn.ReLU(True)

    def forward(self, diff, cat):
        # diff_attn = self.attn_block_diff(diff, cat)
        # cat_attn = self.attn_block_cat(diff, cat)
        fusion_attn = torch.cat([diff, cat], dim=1)
        # fusion_attn = torch.cat([diff, cat], dim=1)

        # Double_Conv add identity process fusion_attn as a Residual function
        DCR_fusion_attn = self.double_conv(fusion_attn) + self.identity(fusion_attn)
        out = self.relu(DCR_fusion_attn)
        return out
