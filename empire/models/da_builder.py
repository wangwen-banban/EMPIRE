import torch
import sys
from pathlib import Path

_models_dir = Path(__file__).resolve().parent
if str(_models_dir) not in sys.path:
    sys.path.insert(0, str(_models_dir))

from depth_anything_3.api import DepthAnything3
import math
import torch.nn as nn
import torch.nn.functional as F


def build_depth_encoder(configs):
    if configs.get("da_weight", None) is not None:
        model = DepthAnything3.from_pretrained(configs["da_weight"])
    elif configs.get("da_name", None) is not None:
        model = DepthAnything3.from_pretrained(configs["da_name"])
    else:
        raise ValueError("Either da_weight or da_name must be provided in configs")  
    
    return model


__all__ = ['MultiHeadAttention', 'ScaledDotProductAttention', 'CAMapper', 'MLPMapper']

class MLPMapper(nn.Module):
    def __init__(self, depth_dim, image_dim, bias=False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(depth_dim, image_dim*2, bias),
            nn.ReLU(inplace=True),
            nn.Linear(image_dim*2, image_dim, bias)
        )
        # self.norm = nn.LayerNorm(image_dim)

    def forward(self, depth_image):
        return self.mlp(depth_image)
    
    def trainable_params(self):
        # total number of parameters
        total_params = sum(p.numel() for p in self.parameters())    
        # count the number of parameters involved in training
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"CAMapper Trainable/Total params: {trainable_params/1e6:.2f}M/{total_params/1e6:.2f}M")
        return trainable_params

class CAMapper(nn.Module):

    def __init__(self, q_dim, k_dim, head_num, layer_num=4, bias=False, zero_init_residual=True):
        super().__init__()
        self.zero_init_residual = zero_init_residual
        # self.attention = MultiHeadAttention(q_dim, k_dim, head_num, bias)
        self.in_mlp = nn.Sequential(
            nn.Linear(q_dim, k_dim, bias),
            nn.ReLU(inplace=True),
            nn.Linear(k_dim, k_dim, bias)
        )
        self.norm1 = nn.LayerNorm(k_dim)
        self.layers = nn.ModuleList([
            MultiHeadAttention(k_dim, k_dim, head_num, bias) for _ in range(layer_num)
        ])
        self.norm2 = nn.LayerNorm(k_dim)
        self.out_mlp = nn.Sequential(
            nn.Linear(k_dim, k_dim, bias),
            nn.ReLU(inplace=True),
            nn.Linear(k_dim, q_dim, bias)
        )

        if self.zero_init_residual:
            nn.init.zeros_(self.out_mlp[-1].weight)
            if self.out_mlp[-1].bias is not None:
                nn.init.zeros_(self.out_mlp[-1].bias)
        
    def forward(self, q, k, v, mask=None):
        residual = q
        q = self.in_mlp(q)
        q = self.norm1(q)
        for layer in self.layers:
            q = q + layer(q, k, v, mask)
        q = self.norm2(q)
        q = self.out_mlp(q)
        return residual + q
    
    def trainable_params(self):
        # total number of parameters
        total_params = sum(p.numel() for p in self.parameters())    
        # count the number of parameters involved in training
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"CAMapper Trainable/Total params: {trainable_params/1e6:.2f}M/{total_params/1e6:.2f}M")
        return trainable_params

class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 q_dim,
                 k_dim,
                 head_num,
                 bias=False,
                 activation=F.relu):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if q_dim % head_num != 0:
            raise ValueError('`q_dim`({}) should be divisible by `head_num`({})'.format(q_dim, head_num))
        if k_dim is None:
            k_dim = q_dim
        if k_dim % head_num != 0:
            raise ValueError('`k_dim`({}) should be divisible by `head_num`({})'.format(k_dim, head_num))
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(q_dim, k_dim, bias)
        self.linear_k = nn.Linear(k_dim, k_dim, bias)
        self.linear_v = nn.Linear(k_dim, k_dim, bias)
        # if q_dim != k_dim:
        self.linear_o = nn.Linear(k_dim, q_dim, bias)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.

        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )
