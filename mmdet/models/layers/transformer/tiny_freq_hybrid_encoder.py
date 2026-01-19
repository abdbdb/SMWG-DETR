# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmengine.model import BaseModule, ModuleList, Sequential
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType
from .metaformer import ASMM, MetaFormer

from timm.models.layers import trunc_normal_
from sapa import sim, atn
from pytorch_wavelets import DWTForward


class RepVGGBlock(BaseModule):
    """RepVGG block is modifided to skip branch_norm.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        stride (int): The stride of the block. Defaults to 1.
        padding (int): The padding of the block. Defaults to 1.
        dilation (int): The dilation of the block. Defaults to 1.
        groups (int): The groups of the block. Defaults to 1.
        padding_mode (str): The padding mode of the block. Defaults to 'zeros'.
        conv_cfg (dict): The config dict for convolution layers.
            Defaults to None.
        norm_cfg (dict): The config dict for normalization layers.
            Defaults to dict(type='BN').
        act_cfg (dict): The config dict for activation layers.
            Defaults to dict(type='ReLU').
        without_branch_norm (bool): Whether to skip branch_norm.
            Defaults to True.
        init_cfg (dict): The config dict for initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 padding_mode: str = 'zeros',
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU'),
                 without_branch_norm: bool = True,
                 init_cfg: OptConfigType = None):
        super(RepVGGBlock, self).__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # judge if input shape and output shape are the same.
        # If true, add a normalized identity shortcut.
        if out_channels == in_channels and stride == 1 and \
                padding == dilation and not without_branch_norm:
            self.branch_norm = build_norm_layer(norm_cfg, in_channels)[1]
        else:
            self.branch_norm = None

        self.branch_3x3 = self.create_conv_bn(
            kernel_size=3,
            dilation=dilation,
            padding=padding,
        )
        self.branch_1x1 = self.create_conv_bn(kernel_size=1)

        self.act = build_activation_layer(act_cfg)

    def create_conv_bn(self,
                       kernel_size: int,
                       dilation: int = 1,
                       padding: int = 0) -> nn.Sequential:
        """Create a conv_bn layer.

        Args:
            kernel_size (int): The kernel size of the conv layer.
            dilation (int, optional): The dilation of the conv layer.
                Defaults to 1.
            padding (int, optional): The padding of the conv layer.
                Defaults to 0.

        Returns:
            nn.Sequential: The created conv_bn layer.
        """
        conv_bn = Sequential()
        conv_bn.add_module(
            'conv',
            build_conv_layer(
                self.conv_cfg,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                dilation=dilation,
                padding=padding,
                groups=self.groups,
                bias=False))
        conv_bn.add_module(
            'norm',
            build_norm_layer(self.norm_cfg, num_features=self.out_channels)[1])

        return conv_bn

    def forward(self, x: Tensor) -> Tensor:
        """1x1 conv + 3x3 conv + identity shortcut.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """

        if self.branch_norm is None:
            branch_norm_out = 0
        else:
            branch_norm_out = self.branch_norm(x)

        out = self.branch_3x3(x) + self.branch_1x1(x) + branch_norm_out

        out = self.act(out)

        return out


class CSPRepLayer(BaseModule):
    """CSPRepLayer.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        num_blocks (int): The number of blocks in the layer. Defaults to 3.
        expansion (float): The expansion of the block. Defaults to 1.0.
        norm_cfg (:obj:`ConfigDict` or dict, optional): The config dict for
            normalization layers. Defaults to dict(type='BN').
        act_cfg (:obj:`ConfigDict` or dict, optional): The config dict for
            activation layers. Defaults to dict(type='SiLU', inplace=True).
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int = 3,
                 expansion: float = 1.0,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='SiLU', inplace=True)):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.bottlenecks = nn.Sequential(*[
            RepVGGBlock(hidden_channels, hidden_channels, act_cfg=act_cfg)
            for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvModule(
                hidden_channels,
                out_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


@MODELS.register_module()
class TinyAwareFrequencyEnhancedHybridEncoder(BaseModule):
    """HybridEncoder.

    Args:
        layer_cfg (:obj:`ConfigDict` or dict): The config dict for the layer.
        projector (:obj:`ConfigDict` or dict, optional): The config dict for
            the projector. Defaults to None.
        num_encoder_layers (int, optional): The number of encoder layers.
            Defaults to 1.
        in_channels (List[int], optional): The input channels of the
            feature maps. Defaults to [512, 1024, 2048].
        feat_strides (List[int], optional): The strides of the feature
            maps. Defaults to [8, 16, 32].
        hidden_dim (int, optional): The hidden dimension of the MLP.
            Defaults to 256.
        use_encoder_idx (List[int], optional): The indices of the encoder
            layers to use. Defaults to [2].
        pe_temperature (int, optional): The temperature of the positional
            encoding. Defaults to 10000.
        expansion (float, optional): The expansion of the CSPRepLayer.
            Defaults to 1.0.
        depth_mult (float, optional): The depth multiplier of the CSPRepLayer.
            Defaults to 1.0.
        norm_cfg (:obj:`ConfigDict` or dict, optional): The config dict for
            normalization layers. Defaults to dict(type='BN').
        act_cfg (:obj:`ConfigDict` or dict, optional): The config dict for
            activation layers. Defaults to dict(type='SiLU', inplace=True).
        eval_size (int, optional): The size of the test image.
            Defaults to None.
    """

    def __init__(self,
                 layer_cfg: ConfigType,
                 projector: OptConfigType = None,
                 num_encoder_layers: int = 1,
                 in_channels: List[int] = [512, 1024, 2048],
                 feat_strides: List[int] = [8, 16, 32],
                 hidden_dim: int = 256,
                 use_encoder_idx: List[int] = [0, 1, 2],
                 pe_temperature: int = 10000,
                 expansion: float = 1.0,
                 depth_mult: float = 1.0,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='SiLU', inplace=True),
                 eval_size=None,
                 token_mixer='ASMM',
                 mixer_cfg=None,
                 ):
        super(TinyAwareFrequencyEnhancedHybridEncoder, self).__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size

        # encoder channel projection
        self.input_proj = ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                ConvModule(
                    in_channel,
                    hidden_dim,
                    kernel_size=1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=None))

        token_map = {"ASMM": ASMM}

        self.encoder = ModuleList([
            MetaFormer(
                num_blocks=num_encoder_layers,
                token_mixer=token_map[token_mixer],
                mixer_cfg=mixer_cfg,
                size=int(100 / (2**i)))
            for i in use_encoder_idx
        ])

        # top-down fpn
        lateral_convs = list()
        fpn_blocks = list()
        lateral_convs_up = list()
        for idx in range(len(in_channels) - 1, 0, -1):
            lateral_convs.append(
                ConvModule(
                    hidden_dim,
                    hidden_dim,
                    1,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            lateral_convs_up.append(
                ConvModule(
                    hidden_dim,
                    hidden_dim,
                    1,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            fpn_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act_cfg=act_cfg,
                    expansion=expansion))
        self.lateral_convs = ModuleList(lateral_convs)
        self.fpn_blocks = ModuleList(fpn_blocks)
        self.lateral_convs_up = ModuleList(lateral_convs_up)

        # bottom-up pan
        downsample_convs = list()
        pan_blocks = list()
        for idx in range(len(in_channels) - 1):
            downsample_convs.append(
                ConvModule(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            pan_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act_cfg=act_cfg,
                    expansion=expansion))
        self.downsample_convs = ModuleList(downsample_convs)
        self.pan_blocks = ModuleList(pan_blocks)

        if projector is not None:
            self.projector = MODELS.build(projector)
        else:
            self.projector = None

        self.up_sample_list = ModuleList()
        for i in range(2):
            up_sample = WGFM(256)
            self.up_sample_list.append(up_sample)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_size[1] // stride, self.eval_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        proj_feats = [
            self.input_proj[i](inputs[i]) for i in range(len(inputs))
        ]

        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                x = proj_feats[enc_ind]
                x = x.permute(0, 2, 3, 1).contiguous()
                x = self.encoder[i](x)
                x = x.permute(0, 3, 1, 2).contiguous()
                proj_feats[enc_ind] = x

        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](
                feat_high)
            inner_outs[0] = feat_high
            upsample_feat = self.up_sample_list[idx - 1](
                self.lateral_convs_up[len(self.in_channels) - 1 - idx](feat_low),
                feat_high)

            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], axis=1))

            inner_outs.insert(0, inner_out)

        # bottom-up pan
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](
                torch.cat([downsample_feat, feat_high], axis=1))
            outs.append(out)

        if self.projector is not None:
            outs = self.projector(outs)

        return tuple(outs)

    @staticmethod
    def build_2d_sincos_position_embedding(w: int,
                                           h: int,
                                           embed_dim=256,
                                           temperature=10000.) -> Tensor:
        """Build 2D sin-cos position embedding.

        Args:
            w (int): The width of the feature map.
            h (int): The height of the feature map.
            embed_dim (int): The dimension of the embedding.
                Defaults to 256.
            temperature (float): The temperature of the position embedding.
                Defaults to 10000.

        Returns:
            Tensor: The position embedding.
        """

        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, \
            ('Embed dimension must be divisible by 4 for '
             '2D sin-cos position embedding')
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([
            torch.sin(out_w),
            torch.cos(out_w),
            torch.sin(out_h),
            torch.cos(out_h)
        ],
                         axis=1)[None, :, :]


def nearest_interpolate(x, scale_factor=2):
    b, h, w, c = x.shape  # channels last
    return x.repeat(1, 1, 1, scale_factor ** 2).reshape(b, h, w, scale_factor, scale_factor, c).permute(
        0, 1, 3, 2, 4, 5).reshape(b, scale_factor * h, scale_factor * w, c)


class WGFM(nn.Module):
    def __init__(self, dim_y, dim_x=None, out_dim=None,
                 q_mode='encoder_only', v_embed=False,
                 up_factor=2, up_kernel_size=5, embedding_dim=64,
                 qkv_bias=True, norm=nn.LayerNorm):
        super().__init__()
        dim_x = dim_x if dim_x is not None else dim_y
        out_dim = out_dim if out_dim is not None else dim_x

        self.up_factor = up_factor
        self.up_kernel_size = up_kernel_size
        self.embedding_dim = embedding_dim

        self.norm_y = norm(dim_y)
        self.norm_x = norm(dim_x)

        self.q_mode = q_mode
        if q_mode == 'encoder_only':
            self.q = nn.Linear(dim_y, embedding_dim, bias=qkv_bias)
        elif q_mode == 'cat':
            self.q = nn.Linear(dim_x + dim_y, embedding_dim, bias=qkv_bias)
        elif q_mode == 'gate':
            self.qy = nn.Linear(dim_y, embedding_dim, bias=qkv_bias)
            self.qx = nn.Linear(dim_x, embedding_dim, bias=qkv_bias)
            self.gate = nn.Linear(dim_x, 1, bias=qkv_bias)
        else:
            raise NotImplementedError

        self.k = nn.Linear(dim_x, embedding_dim, bias=qkv_bias)

        if v_embed or out_dim != dim_x:
            self.v = nn.Linear(dim_x, out_dim, bias=qkv_bias)

        self.dwt = DWTForward(J=1, wave='haar', mode='zero')

        self.channel_compressor = nn.Conv2d(256, embedding_dim, 1)
        self.content_encoder = nn.Conv2d(
            embedding_dim,
            5 ** 2 * 2 * 2,
            3,
            padding=1,
            dilation=1,
            groups=1)

        self.fusion = CSPRepLayer(
            256 * 2,
            256,
            round(1),
            act_cfg=dict(type='SiLU', inplace=True),
            expansion=1)

        self.apply(self._init_weights)

    def high_freq_kernel(self, x, normed_mask, kernel_size, group=1, up=1):
        b, c, h, w = x.shape
        _, m_c, m_h, m_w = normed_mask.shape
        assert m_h == up * h
        assert m_w == up * w
        pad = kernel_size // 2
        pad_x = F.pad(x, pad=[pad] * 4, mode='reflect')
        unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1, padding=0)
        unfold_x = unfold_x.reshape(b, c * kernel_size * kernel_size, h, w)
        unfold_x = F.interpolate(unfold_x, scale_factor=up, mode='bilinear')
        unfold_x = unfold_x.reshape(b, c, kernel_size * kernel_size, m_h, m_w)
        normed_mask = normed_mask.reshape(b, 1, kernel_size * kernel_size, m_h, m_w)
        res = unfold_x * normed_mask
        res = res.sum(dim=2).reshape(b, c, m_h, m_w)
        return res

    def dwt_(self, x):

        yL, yH = self.dwt(x)

        LH = yH[0][:, :, 0, :, :]
        HL = yH[0][:, :, 1, :, :]
        HH = yH[0][:, :, 2, :, :]

        return yL, LH, HL, HH

    def forward(self, y, x):

        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = x.permute(0, 2, 3, 1).contiguous()
        x_ = self.norm_x(x)

        y_ll, y_lh, y_hl, y_hh = self.dwt_(y)

        y_ll = y_ll.permute(0, 2, 3, 1)
        y_ll = self.norm_y(y_ll)

        if self.q_mode == 'encoder_only':
            q = self.q(y_ll)
        elif self.q_mode == 'cat':
            q = self.q(torch.cat([y, nearest_interpolate(x, self.up_factor)], dim=-1))
        elif self.q_mode == 'gate':
            gate = nearest_interpolate(torch.sigmoid(self.gate(x_)), self.up_factor)
            q = gate * self.qy(y) + (1 - gate) * self.qx(nearest_interpolate(x, self.up_factor))
        else:
            raise NotImplementedError

        q = F.interpolate(q.permute(0, 3, 1, 2), scale_factor=2, mode='nearest').permute(0, 2, 3, 1)

        k = self.k(x_)

        x_ll = self.attention(q, k, x).permute(0, 3, 1, 2).contiguous()

        # 处理高频补充高频
        y_h = y_lh + y_hl + y_hh

        com_y = self.channel_compressor(y_h)
        mask = self.content_encoder(com_y)
        mask = self.kernel_normalizer(mask)
        x_h = self.high_freq_kernel(y, mask, 5)
        x_h = y - x_h

        x_fusion = self.fusion(torch.cat([x_ll, x_h], dim=1))

        return x_fusion

    def kernel_normalizer(self, mask: Tensor) -> Tensor:
        mask = F.pixel_shuffle(mask, 2)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(5**2))
        mask = mask.view(n, mask_channel, -1, h, w)

        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_c, h, w).contiguous()

        return mask

    def attention(self, q, k, v):
        attn = F.softmax(sim(q, k, self.up_kernel_size, 1), dim=-1)
        return atn(attn, v, self.up_kernel_size, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)







