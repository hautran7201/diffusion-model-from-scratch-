import math
import torch
from torch import nn 
from torch.nn import functional as F

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embedd: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embedd, 3 * d_embedd, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embedd, d_embedd, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embedd // n_heads

    def forward(self, x: torch.Tensor, casual_mask=False):
        # x: (Batch, Seg_len, Dim)

        input_shape = x.shape
        batch, seg_len, dim = input_shape

        intermim_shape = (batch, seg_len, self.n_heads, self.d_head)

        # (Batch, Seg_len, Dim) -> (Batch, Seg_len, Dim * 3)
        x = self.in_proj(x)

        # (Batch, Seg_len, Dim * 3) -> 3 tensor (Batch, Seg_len, Dim)
        k, q, v = torch.chunk(x, 3, dim=-1)

        # (Batch, Seg_len, Dim) -> (Batch, Seg_len, H, Dim / H) -> (Batch, H, Seg_len, Dim / H)
        q = torch.view(intermim_shape).transpose(1, 2)
        k = torch.view(intermim_shape).transpose(1, 2)        
        v = torch.view(intermim_shape).transpose(1, 2)

        # (Batch, H, Seg_len, Seg_len)
        weight = q @ k.transpose(-1, -2) 

        if casual_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # (Batch, H, Seg_len, Seg_len) @ (Batch, H, Seg_len, Dim / H) -> (Batch, H, Seg_len, Dim / H)
        output = weight @ v 

        # (Batch, H, Seg_len, Dim / H) -> (Batch, Seg_len, H, Dim / H)
        output = output.transpose(1, 2)

        # (Batch, Seg_len, H, Dim / H) -> (Batch, Seg_len, Dim)
        output = output.view(input_shape)

        # (Batch, Seg_len, Dim) -> (Batch, Seg_len, Dim)
        output = self.out_proj(output)

        return output


class CrossAttention(nn.Module):
    def __init__(self, n_head, n_embedd, n_cross, in_proj_bias=False, out_proj_bias=False):
        super().__init__()
        self.q_proj = nn.Linear(n_embedd, n_embedd, bias=in_proj_bias)
        self.k_proj = nn.Linear(n_cross, n_embedd, bias=in_proj_bias)
        self.v_proj = nn.Linear(n_cross, n_embedd, bias=in_proj_bias)
        self.out_proj = nn.Linear(n_embedd, n_embedd, bias=out_proj_bias)
        self.n_head = n_head
        self.d_head = n_embedd // n_head

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: 'Lantent': (Batch, Seg_lenQ, DimQ)
        # y: 'Context': (Batch, Seg_lenKV, DimKV)

        # (Batch, Seg_lenQ, DimQ)
        input_shape = x.shape
        batch, seg_len, dim = input_shape

        interim_shape = (batch, -1, self.n_head, self.d_head)

        # Multiply query key
        # (Batch, Seg_lenQ, DimQ)
        q = self.q_proj(x)
        # (Batch, Seg_lenKV, DimQ)
        k = self.k_proj(y)
        # (Batch, Seg_lenKV, DimQ)
        v = self.v_proj(y)

        # (Batch, Seg_lenQ, DimQ) -> (Batch, Seg_lenQ, H, DimQ / H) -> (Batch, H, Seq_lenQ, DimQ / H)
        q = q.view(interim_shape).transpose(1, 2)
        # (Batch, Seg_lenKV, DimQ) -> (Batch, Seg_lenKV, H, DimQ / H) -> (Batch, H, Seg_lenKV, DimQ / H)
        k = k.view(interim_shape).transpose(1, 2)
        # (Batch, Seg_lenKV, DimQ) -> (Batch, Seg_lenKV, H, DimQ / H) -> (Batch, H, Seg_lenKV, DimQ / H)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch, H, Seq_lenQ, Seg_lenKV)
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.n_head)

        weight = F.softmax(weight, dim=-1)

        # (Batch, H, Seq_lenQ, Seg_lenKV) @ (Batch, H, Seg_lenKV, DimQ / H) -> (Batch, H, Seq_lenQ, DimQ / H)
        output = weight @ v 

        # (Batch, H, Seq_lenQ, DimQ / H) -> (Batch, Seq_lenQ, H, DimQ / H)
        output = output.transpose(1, 2).continuous()

        # (Batch, Seq_lenQ, H, DimQ / H) -> (Batch, Seg_lenQ, DimQ)
        output = output.view(input_shape)

        output = self.out_proj(output)

        return output