import math
import torch
import einops
from torch import nn

class MatrixMultiplication(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        out = a @ b
        return out


# Channel attention block (CAB)
class CAB(nn.Module):
    def __init__(self, in_channel, out_channel=None, ratio=16):
        super(CAB, self).__init__()

        self.in_channels = in_channel
        self.out_channels = out_channel
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channel

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out)


# Spatial attention block (SAB)
class SAB(nn.Module):
    def __init__(self, kernel_size=3):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

# Global Cross Correlation Module(非对称交叉相关性模块)
class GCCM(nn.Module):
    def __init__(
            self,
            in_channel,
            num_heads=8,
            bias=False,
            res_kernel_size=9,
    ):
        super().__init__()
        assert in_channel % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = in_channel // num_heads
        self.scale = head_dim ** -0.5
        self.LinearExpansion = nn.Linear(in_channel, in_channel * 3, bias=bias)
        self.LinearMapping = nn.Linear(in_channel, in_channel)

        self.CrossCorrelation = MatrixMultiplication()
        self.WeightAssignment = MatrixMultiplication()

        self.dconv = nn.Conv2d(
            in_channels=self.num_heads,
            out_channels=self.num_heads,
            kernel_size=(res_kernel_size, 1),
            padding=(res_kernel_size // 2, 0),
            bias=False,
            groups=self.num_heads,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, A, B):
        # Reshape
        H, W = A.shape[2], A.shape[3]
        A = einops.rearrange(A, 'B C H W -> B (H W) C')
        B = einops.rearrange(B, 'B C H W -> B (H W) C')

        # LinearMapping
        N_A, L_A, C_A = A.shape
        N_B, L_B, C_B = B.shape
        Linear_A = (
            self.LinearExpansion(A)
                .reshape(N_A, L_A, 3, self.num_heads, C_A // self.num_heads)
                .permute(2, 0, 3, 1, 4)
        )
        C_1A, C2_A, W_A = Linear_A.unbind(0)
        Linear_B = (
            self.LinearExpansion(B)
                .reshape(N_B, L_B, 3, self.num_heads, C_B // self.num_heads)
                .permute(2, 0, 3, 1, 4)
        )
        C1_B, C2_B, W_B = Linear_B.unbind(0)

        # Norm
        C_1A = C_1A / C_1A.norm(dim=-1, keepdim=True)
        C2_A = C2_A / C2_A.norm(dim=-1, keepdim=True)
        dconv_A = self.dconv(W_A)

        C1_B = C1_B / C1_B.norm(dim=-1, keepdim=True)
        C2_B = C2_B / C2_B.norm(dim=-1, keepdim=True)
        dconv_B = self.dconv(W_B)

        # CrossCorrelation
        CC_matrix_AB = self.CrossCorrelation(C2_B.transpose(-2, -1), C_1A)
        CC_matrix_BA = self.CrossCorrelation(C2_A.transpose(-2, -1), C1_B)

        # Asymmetric Cross Correlation Feature Map
        ACC_A = 0.5 * W_A + 1.0 / math.pi * self.WeightAssignment(W_A, CC_matrix_AB)
        ACC_B = 0.5 * W_B + 1.0 / math.pi * self.WeightAssignment(W_B, CC_matrix_BA)

        ACC_A = ACC_A / ACC_A.norm(dim=-1, keepdim=True)
        ACC_A += dconv_A
        ACC_A = ACC_A.transpose(1, 2).reshape(N_A, L_A, C_A)
        ACC_A = self.LinearMapping(ACC_A)

        ACC_B = ACC_B / ACC_B.norm(dim=-1, keepdim=True)
        ACC_B += dconv_B
        ACC_B = ACC_B.transpose(1, 2).reshape(N_B, L_B, C_B)
        ACC_B = self.LinearMapping(ACC_B)

        # Reshape
        A = einops.rearrange(A, 'B (H W) C -> B C H W', H=H, W=W)
        B = einops.rearrange(B, 'B (H W) C -> B C H W', H=H, W=W)
        ACC_A = einops.rearrange(ACC_A, 'B (H W) C -> B C H W', H=H, W=W)
        ACC_B = einops.rearrange(ACC_B, 'B (H W) C -> B C H W', H=H, W=W)

        A = A + ACC_A
        B = B + ACC_B

        return A, B
