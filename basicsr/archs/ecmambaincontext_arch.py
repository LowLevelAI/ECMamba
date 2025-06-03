import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
# import cv2
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils.img_util import cal_activation, cal_activation2

from typing import Optional, Callable
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat

import DCNv4

def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r

def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)
    

class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)

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

class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)
        self.conv3 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

        self.conv4 = nn.Conv2d(n_fea_out, n_fea_middle, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(n_fea_out, n_fea_middle, kernel_size=1, bias=True)


    def forward(self, img):
        mean_c = img.mean(dim=1).unsqueeze(1)
        
        input = torch.cat([img,mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)

        L_bar = self.conv2(illu_fea)
        R_bar = self.conv3(illu_fea)

        R_prime = img * L_bar + img
        L_prime = img * R_bar + img

        cond_1 = self.conv4(R_prime)
        cond_2 = self.conv5(L_prime) 

        return R_prime, L_prime, cond_1, cond_2, L_bar, R_bar

class EFF(nn.Module):
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
        out = self.net(x)
        return out

class FASS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.defor_conv = RetinexDCT(channels=self.d_inner)

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=1, D, N)

        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x):
        B, L, C = x.shape
        K = 1 
        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x, illu_feat):
        B, H, W, C = x.shape

        x_, offset = self.defor_conv(x, illu_feat)

        ### feature-aware scanning
        shape = (B, C, H, W)
        act_numbers = cal_activation2(shape, offset)
        sorted_counts, sorted_indices = torch.sort(act_numbers.view(B, -1), descending=True)
        x_ = FA_scan(x_, sorted_indices)

        out = self.forward_core(x_).permute(0, 2, 1)
        reversed_indices = index_reverse(sorted_indices)
        out = FA_scan(out, reversed_indices)

        return out

class RetinexDCT(nn.Module):
    def __init__(self, channels, expand=2, groups=4, act_layer='GELU',norm_layer='LN',
                 offset_scale=1.0,
                 dcn_output_bias=False,
                 dw_kernel_size=None):
        super().__init__()
        self.channels = channels
        self.groups = groups

        self.offset_channel = int(groups * 9 * 2)

        self.norm1 = build_norm_layer(channels, 'LN')
        self.norm_illu = build_norm_layer(channels, 'LN')

        cc = int(channels // expand)
        self.illu_proj = nn.Linear(cc, channels, bias=True)
        
        self.dcn = getattr(DCNv4, 'DCNv4')(
            channels=channels*2,
            group=groups,
            offset_scale=offset_scale,
            dw_kernel_size=dw_kernel_size,
            output_bias=dcn_output_bias,
        )


    def forward(self, x, illu_fea):

        x = self.norm1(x).permute(0, 3, 1, 2).contiguous()
        illu_fea = self.norm_illu(self.illu_proj(illu_fea)).permute(0, 3, 1, 2).contiguous()
        y = torch.cat([x, illu_fea], dim=1) #(B,C,H,W)

        B, C, H, W = y.shape
        L = H * W
        y = y.view(B, C, -1).permute(0, 2, 1).contiguous()

        out, offset_mask = self.dcn(y, shape=(H,W))
        x = out[:, :, :self.channels]
        
        offset_mask = offset_mask.permute(0, 3, 1, 2).contiguous()
        offset_mask_ = offset_mask[:, :self.offset_channel, :, :]

        return x, offset_mask_


def FA_scan(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x
    
class RetinexLayer(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            d_conv=3,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.illu_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        self.in_depthconv = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.illu_depthconv = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        
        self.act_fusion = nn.SiLU()

        self.fass2d = FASS2D(d_model=self.d_model, expand=self.expand, d_state=d_state, **kwargs)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
    
    def forward(self, x, illu_feat):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        z = F.silu(z)  #(B,H,W,C)

        x = self.in_depthconv(x.permute(0, 3, 1, 2).contiguous()) #(B,C,H,W)

        illu_feat_ = self.illu_proj(illu_feat)
        illu_feat_ = self.illu_depthconv(illu_feat_.permute(0, 3, 1, 2).contiguous())

        fused_x = x * illu_feat_  #(B,C,H,W)

        x = self.act_fusion(fused_x).permute(0, 2, 3, 1).contiguous()

        y = self.fass2d(x, illu_feat).view(B, H, W, self.d_inner).contiguous() #(B,H,W,C)
        y = self.out_norm(y) #(B,H,W,C)

        out = self.out_proj(y * z)
        return out


class RMB(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.retinexlayer = RetinexLayer(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)

        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.eff = EFF(hidden_dim)



    def forward(self, input, illu_fea):
        
        input = input.permute(0, 2, 3, 1).contiguous()
        x = self.ln_1(input)  #x:[B,H,W,C]
        out_ssm = self.retinexlayer(x, illu_fea.permute(0, 2, 3, 1).contiguous())  #x: [B,H,W,C]
        x = input*self.skip_scale + out_ssm

        x = x*self.skip_scale2 + self.eff(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        out = x.permute(0, 3, 1, 2).contiguous()
        return out

class AMB(nn.Module):
    def __init__(self, m=-0.80):
        super(AMB, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

class ECMM(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=48, level=1, num_blocks=[1, 1, 1], d_state=16):
        super(ECMM, self).__init__()
        self.dim = dim
        self.level = level

        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                RMB(hidden_dim=dim_level),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = RMB(hidden_dim=dim_level)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                #nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                AMB(m=-1.0),
                RMB(hidden_dim=dim_level // 2),
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

    def forward(self, x, illu_fea):
        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for (RMB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = RMB(fea,illu_fea)  # bchw
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea = self.bottleneck(fea,illu_fea)
        tt = fea
        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            #print('decoder', fea.shape)
            fea = Fution(fea, fea_encoder[self.level - 1 - i])
            illu_fea = illu_fea_list[self.level-1-i]
            fea = LeWinBlcok(fea,illu_fea)

        # Mapping
        out = self.mapping(fea) + x

        return out, tt


class RetinexFormer_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, d_state=16, n_feat=40, level=1, num_blocks=[1, 1, 1]):
        super(RetinexFormer_Single_Stage, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)

        dim_L = int(n_feat)

        self.denoiser_r = ECMM(in_dim=in_channels,out_dim=out_channels,dim=n_feat,level=level,num_blocks=num_blocks, d_state=d_state)
        self.denoiser_l = ECMM(in_dim=in_channels,out_dim=out_channels,dim=dim_L,level=level,num_blocks=num_blocks, d_state=d_state)

        self.R_cond = nn.Conv2d(in_channels, n_feat, kernel_size=1, bias=True)
        self.L_cond = nn.Conv2d(in_channels, dim_L, kernel_size=1, bias=True)
    
    def forward(self, img):

        R_prime, L_prime, cond_1, cond_2, L_bar, R_bar = self.estimator(img)
        
        R_restored, tt = self.denoiser_r(R_prime, cond_1)
        L_restored, _ = self.denoiser_l(L_prime, cond_2)

        output_img = R_restored * L_restored 
        return output_img, L_restored, R_restored, R_prime, L_prime, L_bar, R_bar


@ARCH_REGISTRY.register()
class ECMambaIncontext(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, d_state=16, n_feat=32, stage=1, num_blocks=[1,1]):
        super(ECMambaIncontext, self).__init__()
        self.stage = stage

        modules_body = [RetinexFormer_Single_Stage(in_channels=in_channels, out_channels=out_channels, d_state=d_state, n_feat=n_feat, level=1, num_blocks=num_blocks)
                        for _ in range(stage)]
        
        self.body = nn.Sequential(*modules_body)
    
    def forward(self, x):
        out = self.body(x)

        return out
