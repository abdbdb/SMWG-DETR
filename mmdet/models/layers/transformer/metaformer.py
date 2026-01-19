import torch.nn as nn
import torch
import torch.nn.functional as F


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4.0, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = [drop, drop]

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ASMM(nn.Module):  # 不重用系数
    def __init__(self, dim, expansion_ratio=2,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, size=14,
                 num_blocks=8, sparsity_threshold=0.01,
                 hard_thresholding_fraction=1,
                 **kwargs):
        super().__init__()

        visdrone = kwargs.get('visdrone', None)  # 如果没有则返回 None
        if visdrone is None:
            size = [size, size]
        else:
            size = [size, int(size * 1.68)]

        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)

        assert self.med_channels % num_blocks == 0, f"hidden_size {self.med_channels} should be divisble by num_blocks {num_blocks}"

        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.med_channels // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02

        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.w11 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.w22 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.act2 = act2_layer()

        self.act_real = StarReLU()
        self.act_imag = StarReLU()

        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x):

        x = self.pwconv1(x)
        x = self.act1(x)

        x = x.to(torch.float32)
        bias = x
        B, H, W, C = x.shape

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        x = x.reshape(B, self.size, self.filter_size, self.num_blocks, self.block_size)


        o1_real = self.act_real(
            torch.einsum('...bi,bio->...bo', x.real, self.w1[0]) +
            torch.einsum('...bi,bio->...bo', x.imag, self.w1[1]) + self.b1[0]
        )

        o1_imag = self.act_imag(
            torch.einsum('...bi,bio->...bo', x.imag, self.w11[0]) +
            torch.einsum('...bi,bio->...bo', x.real, self.w11[1]) + self.b1[1]
        )

        o2_real = (
                torch.einsum('...bi,bio->...bo', o1_real, self.w2[0]) +
                torch.einsum('...bi,bio->...bo', o1_imag, self.w2[1]) + self.b2[0]
        )

        o2_imag = (
                torch.einsum('...bi,bio->...bo', o1_imag, self.w22[0]) +
                torch.einsum('...bi,bio->...bo', o1_real, self.w22[1]) + self.b2[1]
        )

        o2_real = o2_real * x.real
        o2_imag = o2_imag * x.imag

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = torch.view_as_complex(x)
        x = x.reshape(B, self.size, self.filter_size, C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        x = x + bias

        x = self.act2(x)
        x = self.pwconv2(x)

        return x


class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """

    def __init__(self, dim=256,
                 token_mixer=ASMM, mlp=Mlp,
                 norm_layer=nn.LayerNorm,
                 drop=0.,
                 size=100,
                 mixer_cfg=None
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        if mixer_cfg is None:
            self.token_mixer = token_mixer(dim=dim, size=size)
        else:
            self.token_mixer = token_mixer(dim=dim, size=size, **mixer_cfg)

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)

    def forward(self, x):
        x = x + self.token_mixer(self.norm1(x))

        x = x + self.mlp(self.norm2(x))

        return x


class MetaFormer(nn.Module):
    """
    MetaFormer module containing multiple MetaFormerBlock modules.
    """

    def __init__(self, num_blocks=3, dim=256,
                 token_mixer=ASMM, mlp=Mlp,
                 norm_layer=nn.LayerNorm, drop=0.,
                 size=100, mixer_cfg=None):
        super().__init__()

        # List to store MetaFormerBlock modules
        if mixer_cfg is None:
            mixer_cfg = {}
        self.blocks = nn.ModuleList([
            MetaFormerBlock(dim=dim, token_mixer=token_mixer, mlp=mlp,
                            norm_layer=norm_layer, drop=drop, size=size, mixer_cfg=mixer_cfg)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        # Pass input through each MetaFormerBlock
        for block in self.blocks:
            x = block(x)
        return x
