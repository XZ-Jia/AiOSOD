import torch.nn as nn
import torch
from .pvtv2 import pvt_v2_b0
from .Blocks import TFM
from .Convdecoder import decoder


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.backbone = pvt_v2_b0(args)
        self.TFM = TFM(256, 256) ##6564383
        self.score = nn.Sequential(
            nn.Conv2d(256, 1, 3, 1, 1),
            nn.Conv2d(256, 1, 3, 1, 1),
            nn.Conv2d(256, 1, 3, 1, 1)
            )
    def forward(self, image_Input):
        B, _, _, _ = image_Input.shape
        top_preds = []
        outs, outs_token = self.backbone(image_Input)
        siameae_fea_32, siameae_fea_16, siameae_fea_8, siameae_fea_4 = outs[3],  outs[2],  outs[1],  outs[0]
        fea_32 = self.TFM(outs_token[-1])
        top_preds.append(self.score[0](siameae_fea_32.chunk(2)[0]))
        top_preds.append(self.score[1](siameae_fea_32.chunk(2)[1]))
        top_preds.append(self.score[2](fea_32))
        return fea_32, siameae_fea_32, siameae_fea_16, siameae_fea_8, siameae_fea_4, top_preds

class ImageDepthNet(nn.Module):
    def __init__(self, args):
        super(ImageDepthNet, self).__init__()
        self.Encoder = Encoder(args)
        self.Decoder = decoder()
    def forward(self, image_Input):
        fea_32, siameae_fea_32, siameae_fea_16, siameae_fea_8, siameae_fea_4, top_preds = self.Encoder(image_Input)
        outputs = self.Decoder(fea_32, siameae_fea_32, siameae_fea_16, siameae_fea_8, siameae_fea_4)
        return outputs, top_preds
