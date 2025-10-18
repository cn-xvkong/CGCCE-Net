import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F

from method.main_code.conv import Conv
from .GCCM import GCCM, CAB, SAB
from .Residual import Residual
from .SCE import SCE
from .CFD import Decoder, InteractionDecoder
from .encoder.PVTv2 import pvt_v2_b2, pvt_v2_b1

# Change-Guided Deep Feature Cross-Correlation Enhancement Network for Remote Sensing Change Detection
class CGCCE(nn.Module):
    def __init__(self, num_classes=1, drop=0.2, pvt_pretrain=True):
        super(CGCCE, self).__init__()

        # Transformer Branch with PVTv2 pretraining
        # Params choose b2
        self.backbone = pvt_v2_b2()
        if pvt_pretrain:
            path = 'pvt_v2_b2.pth'
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)

        self.reprocessing = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
        )

        self.GCCM320 = GCCM(in_channel=320)
        self.GCCM512 = GCCM(in_channel=512)

        self.CAB320 = CAB(in_channel=320, out_channel=320)
        self.CAB512 = CAB(in_channel=512, out_channel=512)
        self.SAB = SAB()

        self.SCE1 = SCE(64)
        self.SCE2 = SCE(128)
        self.SCE3 = SCE(320)
        self.SCE4 = SCE(512)

        self.InteractionDecoder1 = InteractionDecoder(512, 512, 512)
        self.InteractionDecoder2 = InteractionDecoder(320, 320, 320)
        self.InteractionDecoder3 = InteractionDecoder(128, 128, 128)
        self.InteractionDecoder4 = InteractionDecoder(64, 64, 64)
        self.Decoder1 = Decoder(512, 320, 320)
        self.Decoder2 = Decoder(320, 128, 128)
        self.Decoder3 = Decoder(128, 64, 64)

        self.re_conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        self.re_conv2 = nn.Sequential(
            nn.Conv2d(640, 320, kernel_size=3, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(True),
        )
        self.re_conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.re_conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.final_process = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.res0 = Residual(64,64)
        self.res1 = Residual(64,128)
        self.res2 = Residual(128,320)
        self.res3 = Residual(320,512)

        self.drop = nn.Dropout2d(drop)

        self.num_images = 0

        self.c1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.c4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        self.c5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
        )
    #
    #     self.s = nn.Sigmoid()
    #
    # def vis_feature(self, feas):
    #     self.num_images += 1
    #     for i, f in enumerate(feas):
    #         f = f[0].cpu().mean(dim=0)
    #         # if i == 4 or i == 5:
    #         #     f = f.view(64, 64)
    #         path = f'test_image/RGB_{self.num_images}_{i}.png'
    #         fig, ax = plt.subplots(1, 1, tight_layout=True)
    #         ax.imshow(f.detach(), cmap='jet')
    #         ax.axis('off')
    #         plt.savefig(path, dpi=200, bbox_inches='tight')
    #         plt.close()

    def forward(self, A, B):
        outputs = []


        # 0-(64,64,64) 1-(128,32,32) 2-(320,16,16) 3-(512,8,8)
        pvtA = self.backbone(A)
        pvtB = self.backbone(B)
        #
        # pvtA[2] = pvtA[2] + self.SAB(pvtA[2] + pvtA[2] * self.CAB320(pvtA[2]))
        # pvtB[2] = pvtB[2] + self.SAB(pvtB[2] + pvtB[2] * self.CAB320(pvtB[2]))
        # pvtA[3] = pvtA[3] + self.SAB(pvtA[3] + pvtA[3] * self.CAB512(pvtA[3]))
        # pvtB[3] = pvtB[3] + self.SAB(pvtB[3] + pvtB[3] * self.CAB512(pvtB[3]))
        #
        # pvtA[2], pvtB[2] = self.GCCM320(pvtA[2], pvtB[2])
        # pvtA[3], pvtB[3] = self.GCCM512(pvtA[3], pvtB[3])


        # Pixel-wise Substraction
        Diff1 = torch.abs(pvtA[0] - pvtB[0])
        Diff2 = torch.abs(pvtA[1] - pvtB[1])
        Diff3 = torch.abs(pvtA[2] - pvtB[2])
        Diff4 = torch.abs(pvtA[3] - pvtB[3])

        # Channel-wise Concatenation
        Cat1 = torch.cat([pvtA[0], pvtB[0]], dim=1)
        Cat2 = torch.cat([pvtA[1], pvtB[1]], dim=1)
        Cat3 = torch.cat([pvtA[2], pvtB[2]], dim=1)
        Cat4 = torch.cat([pvtA[3], pvtB[3]], dim=1)
        Cat1 = self.re_conv4(Cat1)
        Cat2 = self.re_conv3(Cat2)
        Cat3 = self.re_conv2(Cat3)
        Cat4 = self.re_conv1(Cat4)

        # Change-Guided Residual Refinement Branch (CGRR)
        # residual_guided0 = self.res0(Cat1)
        # residual_guided1 = self.res1(residual_guided0, flag=1)
        # residual_guided2 = self.res2(residual_guided1, flag=1)
        # residual_guided3 = self.res3(residual_guided2, flag=1)
        #
        # Cat1 = self.SCE1(Cat1 + residual_guided0)
        # Cat2 = self.SCE2(Cat2 + residual_guided1)
        # Cat3 = self.SCE3(Cat3 + residual_guided2)
        # Cat4 = self.SCE4(Cat4 + residual_guided3)

        # 0-(512,8,8) 1-(320,16,16) 2-(128,32,32) 3-(64,64,64)

        fusion1 = self.InteractionDecoder1(Diff4, Cat4)
        fusion2 = self.InteractionDecoder2(Diff3, Cat3)
        fusion3 = self.InteractionDecoder3(Diff2, Cat2)
        fusion4 = self.InteractionDecoder4(Diff1, Cat1)

        fusion_up_1 = self.Decoder1(fusion1, fusion2)
        fusion_up_2 = self.Decoder2(fusion_up_1, fusion3)
        fusion_up_out = self.Decoder3(fusion_up_2, fusion4)

        fusion_out = F.interpolate(self.final_process(fusion_up_out), scale_factor=4, mode='bilinear')

        return fusion_out
