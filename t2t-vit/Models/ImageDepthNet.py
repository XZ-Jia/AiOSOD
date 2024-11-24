import torch.nn as nn
import torch
from .t2t_vit import T2t_vit_t_14,T2t_vit_12,T2t_vit_10,T2t_vit_7
from .Blocks import TFM
from .Convdecoder import decoder


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.backbone = T2t_vit_10(pretrained=True, args=args)
        self.TFM = TFM(256, 64)
        self.score = nn.Sequential(
            nn.Conv2d(256, 1, 3, 1, 1),
            nn.Conv2d(256, 1, 3, 1, 1),
            nn.Conv2d(64, 1, 3, 1, 1)
            )
    def forward(self, image_Input):
        B, _, _, _ = image_Input.shape
        top_preds = []
        siameae_token_16, siameae_fea_8, siameae_fea_4 = self.backbone(image_Input)
        siameae_fea_16 = siameae_token_16.transpose(1, 2).contiguous().reshape(B, 256, 14, 14)
        fea_16 = self.TFM(siameae_token_16)
        pre_rgb = self.score[0](siameae_fea_16.chunk(2)[0])
        pre_depth = self.score[1](siameae_fea_16.chunk(2)[1])
        pre_rgbd = self.score[2](fea_16)
        top_preds.append(pre_rgb)
        top_preds.append(pre_depth)
        top_preds.append(pre_rgbd)
        return fea_16, siameae_fea_16, siameae_fea_8, siameae_fea_4, top_preds
class ImageDepthNet(nn.Module):
    def __init__(self, args):
        super(ImageDepthNet, self).__init__()
        self.Encoder = Encoder(args)
        self.Decoder = decoder()

    def forward(self, image_Input):
        fea_16, siameae_fea_16, siameae_fea_8, siameae_fea_4, top_preds = self.Encoder(image_Input)
        outputs = self.Decoder(fea_16, siameae_fea_16, siameae_fea_8, siameae_fea_4)
        return outputs, top_preds
