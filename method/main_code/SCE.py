import torch
import torch.nn as nn

# Semantic Cognitive Enhancement
class SCE(nn.Module):
    def __init__(self, channels, kernel_sizes=[3, 5, 7]):
        super(SCE, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels)
        )

        # Multi-scale depthwise convolutions
        self.depthwise_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=k, groups=channels, padding=k // 2),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(channels)
            )
            for k in kernel_sizes
        ])

        # Pointwise convolution
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels * len(kernel_sizes), channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels)
        )

        # Global context
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_layer(x)
        residual = x

        # Multi-scale feature extraction
        multi_scale_features = [branch(x) for branch in self.depthwise_branches]
        x = torch.cat(multi_scale_features, dim=1)

        # Pointwise fusion
        x = self.pointwise(x)

        # Add global context
        gc = self.global_context(residual)
        x = x * gc + residual  # Weighted fusion with residual

        return x


