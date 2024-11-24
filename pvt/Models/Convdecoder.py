import torch
import torch.nn as nn
import torch.nn.functional as F
from .Blocks import DoubleConv, MFFM


class decoder_block1(nn.Module):
    def __init__(self):
        super(decoder_block1, self).__init__()
        self.tran = nn.Sequential(
            nn.Conv2d(256, 160, 3, 1, 1),
            nn.Conv2d(256, 32, 3, 1, 1)
            )
        self.conv = DoubleConv(256, 256)
        self.score = nn.Conv2d(256, 1, 3, 1, 1)
    def forward(self, siameae_fea, fea):
        rgb, depth = siameae_fea.chunk(2)
        fea = self.conv(rgb + fea)
        fea0 = self.tran[0](fea)
        fea1 = self.tran[1](fea)

        return fea0, fea1, self.score(fea)

class decoder_block2(nn.Module):
    def __init__(self):
        super(decoder_block2, self).__init__()
        self.tran = nn.Sequential(
            nn.Conv2d(160, 64, 3, 1, 1),
            nn.Conv2d(160, 32, 3, 1, 1)
            )
        self.conv = DoubleConv(160, 160)
        self.score = nn.Conv2d(160, 1, 3, 1, 1)
    def forward(self, siameae_fea, fea):
        rgb, depth = siameae_fea.chunk(2)
        fea = F.interpolate(fea, siameae_fea.size()[2:], mode='bilinear', align_corners=True)
        fea = self.conv(rgb + fea)
        fea0 = self.tran[0](fea)
        fea1 = self.tran[1](fea)
        return fea0, fea1, self.score(fea)

class decoder_block3(nn.Module):
    def __init__(self):
        super(decoder_block3, self).__init__()
        self.tran = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv = DoubleConv(64, 64)
        self.score = nn.Conv2d(64, 1, 3, 1, 1)
    def forward(self, siameae_fea, fea):
        rgb, depth = siameae_fea.chunk(2)
        fea = F.interpolate(fea, siameae_fea.size()[2:], mode='bilinear', align_corners=True)
        fea = self.conv(rgb + fea)
        fea1 = self.tran(fea)
        return fea1, self.score(fea)
class decoder_block4(nn.Module):
    def __init__(self):
        super(decoder_block4, self).__init__()
        self.conv = DoubleConv(32, 32)
        self.score = nn.Conv2d(32, 1, 3, 1, 1)
    def forward(self, siameae_fea, fea):
        rgb, depth = siameae_fea.chunk(2)
        fea = F.interpolate(fea, siameae_fea.size()[2:], mode='bilinear', align_corners=True)
        fea = self.conv(rgb + fea)
        return fea, self.score(fea)
class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.Block = nn.Sequential(
            decoder_block1(),
            decoder_block2(),
            decoder_block3(),
            decoder_block4()
        )
        self.MFFM = MFFM(32)
        self.score = nn.Conv2d(32, 1, 3, 1, 1)
    def forward(self, fea_32, siameae_fea_32, siameae_fea_16, siameae_fea_8, siameae_fea_4):

        sal_fea_32, sal_fea_32_C32, mask_1_32 = self.Block[0](siameae_fea_32, fea_32)
        sal_fea_16, sal_fea_16_C32, mask_1_16 = self.Block[1](siameae_fea_16, sal_fea_32)
        sal_fea_8_C32, mask_1_8 = self.Block[2](siameae_fea_8, sal_fea_16)
        sal_fea_4_C32, mask_1_4 = self.Block[3](siameae_fea_4, sal_fea_8_C32)

        mlf_fea_8 = F.interpolate(sal_fea_8_C32, sal_fea_4_C32.size()[2:], mode='bilinear', align_corners=True)
        mlf_fea_16  = F.interpolate(sal_fea_16_C32, sal_fea_4_C32.size()[2:], mode='bilinear', align_corners=True)
        mlf_fea_32  = F.interpolate(sal_fea_32_C32, sal_fea_4_C32.size()[2:], mode='bilinear', align_corners=True)
        mask_1_1 = self.score(self.MFFM(torch.cat((sal_fea_4_C32, mlf_fea_8, mlf_fea_16, mlf_fea_32),dim=1)))

        mask_1_1 = F.interpolate(mask_1_1, (224, 224), mode='bilinear', align_corners=True)
        return [mask_1_32, mask_1_16, mask_1_8, mask_1_4, mask_1_1]
