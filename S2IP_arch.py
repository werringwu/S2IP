import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import os

import json
import shutil


class ResBlock(nn.Module):
    """ """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.downsample = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x += residual
        return F.relu(x)

class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.block1 = ResBlock(64, 64)
        self.block2 = ResBlock(64, 128)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.stem(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
#from basicsr.utils.registry import ARCH_REGISTRY
import math

# import cv2


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# input [bs,28,256,310]  output [bs, 28, 256, 256]
def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]


class IGCE(nn.Module):  # Illumination_Estimator 
    def __init__(self, n_fea_middle=40):
        super().__init__()

        self.conv1 = nn.Conv2d(4, n_fea_middle, 1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2,
            bias=True, groups=n_fea_middle
        )

        self.conv_dark = nn.Conv2d(n_fea_middle, 1, 3, padding=1, bias=True)
        self.conv_color_local = nn.Conv2d(n_fea_middle, 3, 3, padding=1, bias=True)

        #  global gain
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_wb = nn.Conv2d(n_fea_middle, 3, 1, bias=True)
        # [alpha, gamma, beta_color]
        self.fc_curve = nn.Conv2d(n_fea_middle, 3, 1, bias=True)

    def forward(self, img):
        """
        img: (B,3,H,W)
        return:
          input_img : (B,3,H,W) 
          prior_fea  : (B,Cm,H,W) 
        """
        B, C, H, W = img.shape

        mean_c = img.mean(dim=1, keepdim=True)           # (B,1,H,W)
        x = torch.cat([img, mean_c], dim=1)              # (B,4,H,W)
        x = self.conv1(x)
        prior_fea = self.depth_conv(x)                    # (B,Cm,H,W)

        dark = torch.sigmoid(self.conv_dark(prior_fea))         # (B,1,H,W)
        color_local = torch.tanh(self.conv_color_local(prior_fea))  # (B,3,H,W), ∈[-1,1]

        g = self.global_pool(prior_fea)                  # (B,Cm,1,1)
        wb_raw = self.fc_wb(g)                          # (B,3,1,1)
        wb = 1.0 + 0.5 * torch.tanh(wb_raw)             # [0.5, 1.5]

        curve_raw = self.fc_curve(g)                    # (B,3,1,1)
        alpha = 3.0 * torch.sigmoid(curve_raw[:, 0:1])       # (B,1,1,1) ∈ (0,3)
        gamma = 1.0 + 3.0 * torch.sigmoid(curve_raw[:, 1:2]) # ∈ (1,4)
        beta_color = 0.5 * torch.sigmoid(curve_raw[:, 2:3])  # ∈ (0,0.5)

        img_wb = torch.clamp(img * wb, 0.0, 1.0)        # (B,3,H,W)

        gain = 1.0 + alpha * (dark ** gamma)            # (B,1,1,1) 与 (B,1,H,W) 
        gain = gain.expand(-1, 1, H, W)                 # (B,1,H,W)

        color_scale_local = 1.0 + beta_color * color_local  # (B,3,H,W)

        input_img = img_wb * gain * color_scale_local
        input_img = torch.clamp(input_img, 0.0, 1.0)

        return input_img, prior_fea

class PGA_Layer(nn.Module): # Prior-Guided Attention
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, prior_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        prior_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        prior_attn = prior_fea_trans  # prior_fea: b,c,h,w -> b,h,w,c
        q, k, v, prior_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, prior_attn.flatten(1, 2)))
        v = v * prior_attn
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class BKE(nn.Module): #BlurKernelEstimator
    def __init__(self, dim, down_scale=2):
        super().__init__()
        self.dim = dim
        self.down_scale = down_scale
        self.downsample = nn.Conv2d(dim, dim, kernel_size=down_scale, stride=down_scale, padding=0, bias=False)
        self.blur_kernel = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.upsample = nn.ConvTranspose2d(dim, dim, kernel_size=down_scale, stride=down_scale, padding=0,
                                           output_padding=0, bias=False)
        nn.init.dirac_(self.blur_kernel.weight, groups=dim)  
        nn.init.kaiming_normal_(self.downsample.weight)
        nn.init.kaiming_normal_(self.upsample.weight)

    def forward(self, fea):
        fea_down = self.downsample(fea)
        fea_blur = self.blur_kernel(fea_down)
        fea_blur_up = self.upsample(fea_blur)
        return fea_blur_up



class CADC(nn.Module): # Context-Aware Detail Calibration
    def __init__(self, dim):
        super().__init__()

        self.denoise = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
            LayerNorm2d(dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)  
        )

    def forward(self, fea_ori, fea_blur):
        diff = fea_ori - fea_blur
        diff_clamp = torch.clamp(diff, min=0.0)
        diff_clean = self.denoise(diff_clamp)
        return diff_clean


class SDM(nn.Module): #Spatial Detail Modulator
    def __init__(self, dim):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 4, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.ones(1))  

    def forward(self, fea_blur, diff_clean):
        attn_weight = self.attn(diff_clean)
        fea_restored = fea_blur + self.gamma * diff_clean * attn_weight
        return fea_restored

class FreMLP(nn.Module):

    def __init__(self, nc, expand=1):
        super(FreMLP, self).__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))
        self.process2 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        pha = self.process2(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # Haar transform is fixed

    def forward(self, x):
        """
        Input: [B, C, H, W]
        Output: [B, 4*C, H/2, W/2]  (Channel order: LL, LH, HL, HH)
        """
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        x_LL = x1 + x2 + x3 + x4
        x_LH = -x1 - x2 + x3 + x4
        x_HL = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return torch.cat((x_LL, x_LH, x_HL, x_HH), 1)
        # return x_LL,x_LH,x_HL,x_HH


class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        """
        Input: [B, 4*C, H/2, W/2] (Channel order: LL, LH, HL, HH)
        Output: [B, C, H, W]
        """
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        out_batch, out_channel, out_height, out_width = in_batch, int(
            in_channel // (r ** 2)), r * in_height, r * in_width

        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

        h = torch.zeros((out_batch, out_channel, out_height, out_width), device=x.device, dtype=x.dtype)

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h


class WFFT(nn.Module):
    def __init__(self, c):
        super(WFFT, self).__init__()
        self.dwt = DWT()
        self.idwt = IDWT()

        # We process the High Frequency sub-bands (LH, HL, HH) which are 3*C channels
        # LL (Low Frequency) is usually preserved or lightly processed.
        # Here we focus on refining details.
        self.ln_LL = LayerNorm2d(c)
        self.ln_LH = LayerNorm2d(c)
        self.ln_HL = LayerNorm2d(c)
        self.ln_HH = LayerNorm2d(c)
        self.process_LL = nn.Sequential(
            nn.Conv2d(c * 2, c * 2, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c * 2, c * 2, 3, 1, 1, groups=c * 2),  # Depthwise for efficiency
            nn.Conv2d(c * 2, c * 1, 1, 1, 0)
        )
        self.process_LH = nn.Sequential(
            nn.Conv2d(c * 2, c * 2, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c * 2, c * 2, 3, 1, 1, groups=c * 2),  # Depthwise for efficiency
            nn.Conv2d(c * 2, c * 1, 1, 1, 0)
        )
        self.process_HL = nn.Sequential(
            nn.Conv2d(c * 2, c * 2, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c * 2, c * 2, 3, 1, 1, groups=c * 2),  # Depthwise for efficiency
            nn.Conv2d(c * 2, c * 1, 1, 1, 0)
        )
        self.process_HH = nn.Sequential(
            nn.Conv2d(c * 2, c * 2, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c * 2, c * 2, 3, 1, 1, groups=c * 2),  # Depthwise for efficiency
            nn.Conv2d(c * 2, c * 1, 1, 1, 0)
        )
        self.LL_FreMLP = FreMLP(nc=c, expand=2)
        self.LH_FreMLP = FreMLP(nc=c, expand=2)
        self.HL_FreMLP = FreMLP(nc=c, expand=2)
        self.HH_FreMLP = FreMLP(nc=c, expand=2)

        # Lightweight fusion of LL and refined Highs
        self.fusion = nn.Sequential(
            nn.Conv2d(c * 4, c * 4, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c * 4, c * 4, 1, 1, 0),
        )
        self.pixs = nn.PixelShuffle(2)

    def forward(self, x):
        # 1. Decompose
        x_dwt = self.dwt(x)  # [B, 4C, H/2, W/2]
        B, C4, H, W = x_dwt.shape
        C = C4 // 4

        # 2. Split into LL (Approximation) and High Frequencies (Details)
        x_LL = x_dwt[:, :C, :, :]
        x_LH = x_dwt[:, C:2 * C, :, :]
        x_HL = x_dwt[:, 2 * C:3 * C, :, :]
        x_HH = x_dwt[:, 3 * C:, :, :]  # [B, 3C, H/2, W/2]

        x_LL_ln = self.ln_LL(x_LL)
        x_LL_fft = self.LL_FreMLP(x_LL_ln) * x_LL
        x_LL_refined = torch.cat([x_LL_fft, x_LL], dim=1)
        x_LL_refined = self.process_LL(x_LL_refined) + x_LL

        x_LH_ln = self.ln_LH(x_LH)
        x_LH_fft = self.LH_FreMLP(x_LH_ln) * x_LH
        x_LH_refined = torch.cat([x_LH_fft, x_LH], dim=1)
        x_LH_refined = self.process_LH(x_LH_refined) + x_LH

        x_HL_ln = self.ln_HL(x_HL)
        x_HL_fft = self.HL_FreMLP(x_HL_ln) * x_HL
        x_HL_refined = torch.cat([x_HL_fft, x_HL], dim=1)
        x_HL_refined = self.process_HL(x_HL_refined) + x_HL

        x_HH_ln = self.ln_HH(x_HH)
        x_HH_fft = self.HH_FreMLP(x_HH_ln) * x_HH
        x_HH_refined = torch.cat([x_HH_fft, x_HH], dim=1)
        x_HH_refined = self.process_HH(x_HH_refined) + x_HH

        # 3. Enhance High Frequencies (De-blurring/De-noising happens here)
        # x_High_refined = self.process_high(x_High) + x_High

        # 4. Recombine and Fuse
        x_cat = torch.cat([x_LL_refined, x_LH_refined, x_HL_refined, x_HH_refined], dim=1)
        x_cat = self.fusion(x_cat) # ablation

        # 5. Reconstruct
        x_out = self.pixs(x_cat) # ablation
        #x_out = self.idwt(x_cat)
        return x_out


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=3, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.conv_to_target_channels = nn.Conv2d(channel, channel // 2, 1)

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x

        x = self.conv_to_target_channels(x)
        return x


class ISSM_Module(nn.Module): #Intra-Scale Spectral Modulation
    def __init__(self, in_channels=3, out_channels=3):
        super(ISSM_Module, self).__init__()
        self.wfft = WFFT(in_channels)
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        out = self.wfft(x) * self.gamma + x

        return out

class DPMB(nn.Module): #Prior-Guided Attention Block  Degradation-Prior Modulated Block 
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
            en=True,
    ):
        super().__init__()
        self.en = en
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PGA_Layer(dim=dim, dim_head=dim_head, heads=heads),
                ISSM_Module(in_channels=dim,out_channels=dim),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, prior_fea):
        """
        x: [b,c,h,w]
        prior_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn,wtfft, ff) in self.blocks:
            if self.en:
                x = wtfft(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                x = attn(x, prior_fea_trans=prior_fea.permute(0, 2, 3, 1)) + x
            else:
                x = attn(x, prior_fea_trans=prior_fea.permute(0, 2, 3, 1)) + x
                x = wtfft(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class RestorationTrunk(nn.Module): #
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4],multiOut=False):
        super(RestorationTrunk, self).__init__()
        self.dim = dim
        self.level = level
        self.multiOut = multiOut
        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                DPMB(
                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim,en=True),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.ConvTranspose2d(dim_level * 2, dim_level, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),

            ]))
            dim_level *= 2

        # Bottleneck
        # self.bottleneck = DPMB(
        #     dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        self.bottleneck = nn.ModuleList([
            BKE(dim=dim_level, down_scale=2),
            CADC(dim=dim_level),
            SDM(dim=dim_level),
            DPMB(dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1],en=True),
        ])
        self.mapping_D = nn.Conv2d(dim_level, out_dim, 3, 1, 1, bias=False)
        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                BKE(dim=dim_level // 2, down_scale=2),
                CADC(dim=dim_level // 2),
                SDM(dim=dim_level // 2),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                DPMB(
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim,en=False),
            ]))
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, prior_fea):
        """
        x:          [b,c,h,w]         x是feature, 不是image
        prior_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        prior_fea_list = []
        for (DPMB, FeaDownSample, IlluFeaDownsample, FeaUpSample, rsDownSample) in self.encoder_layers:
            fea0 = fea
            fea = DPMB(fea, prior_fea)  # bchw
            prior_fea_list.append(prior_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            prior_fea = IlluFeaDownsample(prior_fea)
            up_fea = FeaUpSample(fea)
            difffea = torch.clamp(fea0 - up_fea, min=0.0)
            fea = rsDownSample(difffea) + fea
        # Bottleneck
        fea_blur = self.bottleneck[0](fea)
        fea_guide = self.bottleneck[1](fea, fea_blur)
        fea_clean = self.bottleneck[2](fea_blur, fea_guide)
        fea = self.bottleneck[3](fea_clean, prior_fea)
        out_D = self.mapping_D(fea)
        # Decoder
        for i, (FeaUpSample, lrker, diffnoise, guidefusion, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)

            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))

            fea_blur = lrker(fea)
            fea_guide = diffnoise(fea, fea_blur)
            fea = guidefusion(fea_blur, fea_guide)
            prior_fea = prior_fea_list[self.level - 1 - i]
            fea = LeWinBlcok(fea, prior_fea)

        # Mapping
        out = self.mapping(fea) + x
        if self.multiOut:
            return out
        else:
            return out

class S2IP_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=40, level=2, num_blocks=[1, 2, 3],multiOut=False):
        super(S2IP_Stage, self).__init__()
        self.estimator = IGCE(n_feat)
        self.restoration = RestorationTrunk(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level,
                                 num_blocks=num_blocks,multiOut=multiOut)  


    def forward(self, img):
        # img:        b,c=3,h,w

        # prior_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        input_img, prior_fea= self.estimator(img)

        output_img = self.restoration(input_img, prior_fea)

        return output_img

class S2IP(nn.Module): 
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=[1, 3, 5],level=3):
        super(S2IP, self).__init__()
        self.stage = stage

        modules_body = [
            S2IP_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=level,
                                       num_blocks=num_blocks,multiOut=True)
            for _ in range(stage)]

        self.body = nn.Sequential(*modules_body)

    def forward(self, x,side_loss = False):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        out = self.body(x)
        if side_loss:
            return out
        else:
            return out