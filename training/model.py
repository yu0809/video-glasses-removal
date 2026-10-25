"""
Frame segmentation network
==========================
U-Net with a timm pretrained encoder (resnet34 by default; efficientnet_b3 / convnext_tiny or any features_only model work).
The decoder upsamples all the way to full resolution (thin frames are only 1-3 px wide), and a shallow full-resolution
branch (3x3 conv features of the input image) is concatenated before the last layer.
Output: 2-channel logits, 0 = frame, 1 = lens (auxiliary).
Input: pipeline face crop (BGR -> RGB, /255, ImageNet mean/std), 384x384 by default.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def conv_bn(cin, cout, k=3):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UpBlock(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.c1 = conv_bn(cin + cskip, cout)
        self.c2 = conv_bn(cout, cout)

    def forward(self, x, skip=None):
        size = skip.shape[-2:] if skip is not None else (x.shape[-2] * 2, x.shape[-1] * 2)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], 1)
        return self.c2(self.c1(x))


class FrameSegNet(nn.Module):
    def __init__(self, encoder="resnet34", pretrained=True, dec_ch=(256, 128, 64, 48, 32), hires_ch=24):
        super().__init__()
        import timm
        self.encoder_name = encoder
        self.enc = timm.create_model(encoder, features_only=True, pretrained=pretrained)
        chs = self.enc.feature_info.channels()          # shallow to deep
        red = self.enc.feature_info.reduction()
        self.reductions = red
        # from the deepest level upwards, concatenating shallower features; the last level is upsampled to full resolution
        ups, cin = [], chs[-1]
        skips = list(chs[:-1])[::-1]
        for i, skip in enumerate(skips):
            ups.append(UpBlock(cin, skip, dec_ch[min(i, len(dec_ch) - 1)]))
            cin = dec_ch[min(i, len(dec_ch) - 1)]
        self.ups = nn.ModuleList(ups)
        self.cin_last = cin
        self.hires = nn.Sequential(conv_bn(3, hires_ch), conv_bn(hires_ch, hires_ch))
        self.final = nn.Sequential(conv_bn(cin + hires_ch, 32), nn.Conv2d(32, 2, 1))

    def forward(self, x):
        feats = self.enc(x)
        y = feats[-1]
        for up, skip in zip(self.ups, feats[:-1][::-1]):
            y = up(y, skip)
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        y = torch.cat([y, self.hires(x)], 1)
        return self.final(y)


def build_model(cfg, pretrained=True):
    return FrameSegNet(encoder=cfg.get("encoder", "resnet34"), pretrained=pretrained)


def normalize_batch(x_uint8_bgr):
    """(B,H,W,3) uint8 BGR tensor -> (B,3,H,W) normalised float."""
    x = x_uint8_bgr[..., [2, 1, 0]].permute(0, 3, 1, 2).float() / 255
    m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return ((x - m) / s).contiguous()      # a non-contiguous input makes the first conv backward >4x slower on MPS
