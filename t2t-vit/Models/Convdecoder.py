import torch
import torch.nn as nn
import torch.nn.functional as F
from .Blocks import DoubleConv, MFFM


class decoder_block(nn.Module):
    def __init__(self, tran = False):
        super(decoder_block, self).__init__()
        if tran:
            self.tran = nn.Conv2d(256, 64, 3, 1, 1)
        self.conv = DoubleConv(64, 64)
        self.score = nn.Conv2d(64, 1, 3, 1, 1)
    def forward(self, siameae_fea, fea):
        rgb, depth = siameae_fea.chunk(2)
        if fea.size()[2:] != siameae_fea.size()[2:]:
            fea = F.interpolate(fea, siameae_fea.size()[2:], mode='bilinear', align_corners=True)
        else:
            rgb = self.tran(rgb)
        fea = self.conv(rgb+fea)
        return fea, self.score(fea)

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.score = nn.Conv2d(64, 1, 3, 1, 1)
        self.Block = nn.Sequential(
            decoder_block(True),
            decoder_block(),
            decoder_block()
        )
        self.MFFM = MFFM()
    def forward(self, fea_1_16, siameae_fea_16, siameae_fea_8, siameae_fea_4):

        sal_fea_16, mask_1_16 = self.Block[0](siameae_fea_16, fea_1_16)
        sal_fea_8, mask_1_8 = self.Block[1](siameae_fea_8, sal_fea_16)
        sal_fea_4, mask_1_4 = self.Block[2](siameae_fea_4, sal_fea_8)

        sal_fea_8 = F.interpolate(sal_fea_8, sal_fea_4.size()[2:], mode='bilinear', align_corners=True)
        sal_fea_16  = F.interpolate(sal_fea_16, sal_fea_4.size()[2:], mode='bilinear', align_corners=True)
        mask_1_1 = self.score(self.MFFM(torch.cat((sal_fea_4, sal_fea_8, sal_fea_16),dim=1)))

        mask_1_1 = F.interpolate(mask_1_1, (224, 224), mode='bilinear', align_corners=True)
        return [mask_1_16, mask_1_8, mask_1_4, mask_1_1]
