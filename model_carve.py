import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional # Optional类型提示表示一个值可以是某种特定类型或者是None

"""
随机深度（Stochastic Depth）: 对输入数据进行随机丢弃，以增强模型的泛化能力；
"""
def drop_path_f(x, drop_prob: float = 0., training: bool = False): # drop_prob：参数名，float：期望的数据类型，0：默认值
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # 生成随机决策的 random_tensor 的形状 shape
    # shape是元组类型，(1,)括号中带逗号的写法表示这是个元组类型数据，x.ndim获取x张量的维度，比如说3，(1,)*(3-1)表示将元组（1,)重复2次，即(1,1)，而不是将元组中的元素乘以2
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # shape值为(batch, 1, 1, 1)
    # torch.rand()创建一个指定形状的张量,随机初始化为在区间 [0, 1) 上均匀分布，浮点数与张量相加，会自动进行广播机制得到一个张量
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor # 运用两次广播机制
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

"""
Step/1
"""
class WindowAttention(nn.Moudle):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
        It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0

    """
    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True,
                 attn_drop=0.,
                 proj_drop=0.
                 ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)  # [2*Mh-1 * 2*Mw-1, nH]
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing = "ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1) # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # [2, Mh*Mw, Mh*Mw]
        relation_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]




"""
Step/0
"""
class SwinTransformerBlock(nn.Module):
    r"""
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        drop (float, optional): Dropout rate. Default: 0.0.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0.
        drop_path (float, optional): Stochastic depth rate. Default: 0.0.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, num_heads,
                 window_size = 7,
                 shift_size = 0,
                 mlp_ratio = 4.,
                 qkv_bias = True,
                 drop = 0.,
                 attn_drop = 0.,
                 drop_path = 0.,
                 act_layer = nn.GELU,
                 norm_layer = nn.LayerNorm
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim,
                                    window_size = (self.window_size, self.window_size),
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    attn_drop=attn_drop,
                                    proj_drop=drop
                                    )














