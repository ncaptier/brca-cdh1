"""
In parts from https://github.com/KatherLab/STAMP/blob/main/stamp/modeling/marugoto/transformer/TransMIL.py
"""

import torch
from torch import nn
from einops import repeat
from stamp.modeling.marugoto.transformer.TransMIL import FeedForward
from .attention_dist import RelationAwareMultiHeadAttention


class AttentionDist(nn.Module):
    def __init__(self, dim, heads=8, k=10, norm_layer=nn.LayerNorm):
        super().__init__()
        self.heads = heads
        self.norm = norm_layer(dim)
        self.mhsa = RelationAwareMultiHeadAttention(dim, heads, k)

    def forward(self, x, coord, mask=None):
        if mask is not None:
            mask = mask.repeat(self.heads, 1, 1)

        x = self.norm(x)
        attn_output = self.mhsa(x, x, x, coord, mask=mask)
        return attn_output


class TransformerDist(nn.Module):
    def __init__(self, dim, depth, heads, k, mlp_dim, norm_layer=nn.LayerNorm, dropout=0.):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                AttentionDist(dim, heads=heads, k=k, norm_layer=norm_layer),
                FeedForward(dim, mlp_dim, norm_layer=norm_layer, dropout=dropout)
            ]))
        self.norm = norm_layer(dim)

    def forward(self, x, coord, mask=None):
        for attn, ff in self.layers:
            x_attn = attn(x, coord, mask=mask)
            x = x_attn + x
            x = ff(x) + x
        return self.norm(x)


class TransMILDist(nn.Module):
    def __init__(self, *,
                 num_classes: int, input_dim: int = 768, dim: int = 512,
                 depth: int = 2, heads: int = 8, k: int = 10, mlp_dim: int = 2048,
                 pool: str = 'cls', dropout: int = 0., emb_dropout: int = 0.
                 ):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.cls_token = nn.Parameter(torch.randn(dim))

        self.fc = nn.Sequential(nn.Linear(input_dim, dim, bias=True), nn.GELU())
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerDist(dim, depth, heads, k, mlp_dim, nn.LayerNorm, dropout)

        self.pool = pool
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, coord, lens):
        b, n, d = x.shape

        # map input sequence to latent space of TransMIL
        x = self.dropout(self.fc(x))

        add_cls = self.pool == 'cls'
        if add_cls:
            cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            lens = lens + 1  # account for cls token

        mask = None
        if torch.amin(lens) != torch.amax(lens) and False:
            mask = torch.arange(0, n + add_cls, dtype=torch.int32, device=x.device).repeat(b, 1) < lens[..., None]
            mask = (~mask[:, None, :]).repeat(1, (n + add_cls), 1)

        x = self.transformer(x, coord, mask)

        if mask is not None and self.pool == 'mean':
            x = torch.cumsum(x, dim=1)[torch.arange(b), lens - 1]
            x = x / lens[..., None]
        else:
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)
