import torch.nn as nn
import torch
from .Swin_Transformer import SwinTransformer
from .Blocks import TFM
from .Convdecoder import decoder


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.backbone = SwinTransformer()
        self.load_pre(args.pretrained_model)
        self.TFM = TFM(768, 64)
        self.tran = nn.Sequential(
            nn.Conv2d(96, 64, 3, 1, 1),
            nn.Conv2d(192, 64, 3, 1, 1),
            nn.Conv2d(384, 64, 3, 1, 1),
            nn.Conv2d(768, 64, 3, 1, 1)
            )
        self.score = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Conv2d(64, 1, 3, 1, 1)
            )
    def forward(self, image_Input):
        B, _, _, _ = image_Input.shape
        top_preds = []
        outs, outs_token = self.backbone(image_Input)
        siameae_fea_32, siameae_fea_16, siameae_fea_8, siameae_fea_4 = outs[3],  outs[2],  outs[1],  outs[0]
        siameae_fea_4 = self.tran[0](siameae_fea_4)
        siameae_fea_8 = self.tran[1](siameae_fea_8)
        siameae_fea_16 = self.tran[2](siameae_fea_16)
        siameae_fea_32 = self.tran[3](siameae_fea_32)
        fea_32 = self.TFM(outs_token[0])
        top_preds.append(self.score[0](siameae_fea_32.chunk(2)[0]))
        top_preds.append(self.score[1](siameae_fea_32.chunk(2)[1]))
        top_preds.append(self.score[2](fea_32))
        return fea_32, siameae_fea_32, siameae_fea_16, siameae_fea_8, siameae_fea_4, top_preds
    def load_pre(self, pre_model=''):
        self.backbone.load_state_dict(torch.load(pre_model)['model'],strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")

class ImageDepthNet(nn.Module):
    def __init__(self, args):
        super(ImageDepthNet, self).__init__()
        self.Encoder = Encoder(args)
        self.Decoder = decoder()

    def forward(self, image_Input):
        fea_32, siameae_fea_32, siameae_fea_16, siameae_fea_8, siameae_fea_4, top_preds = self.Encoder(image_Input)
        outputs = self.Decoder(fea_32, siameae_fea_32, siameae_fea_16, siameae_fea_8, siameae_fea_4)

        return outputs, top_preds
