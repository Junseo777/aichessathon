from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

NUM_PLANES = 21
NUM_MOVES = 4672


@dataclass
class Config:
    d_model: int = 96
    n_heads: int = 4
    n_blocks: int = 12
    ffn_mult: int = 4
    value_hidden: int = 128


def _rel_index() -> torch.Tensor:
    idx = torch.zeros(64, 64, dtype=torch.long)
    for i in range(64):
        for j in range(64):
            dr = (i // 8) - (j // 8) + 7
            df = (i % 8) - (j % 8) + 7
            idx[i, j] = dr * 15 + df
    return idx


class GeoSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.rel_bias = nn.Parameter(torch.zeros(n_heads, 15 * 15))
        self.register_buffer("rel_idx", _rel_index(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n, d = x.shape
        qkv = self.qkv(x).reshape(batch, n, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        bias = self.rel_bias[:, self.rel_idx].unsqueeze(0).to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.proj(out.transpose(1, 2).reshape(batch, n, d))


class Block(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = GeoSelfAttention(cfg.d_model, cfg.n_heads)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        hidden = cfg.ffn_mult * cfg.d_model
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, hidden), nn.GELU(), nn.Linear(hidden, cfg.d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class ChessNet(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Conv2d(NUM_PLANES, cfg.d_model, kernel_size=1, bias=False)
        self.embed_bn = nn.BatchNorm2d(cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 64, cfg.d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_blocks))
        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.policy_head = nn.Linear(cfg.d_model, NUM_MOVES // 64)
        self.value_fc1 = nn.Linear(cfg.d_model, cfg.value_hidden)
        self.value_fc2 = nn.Linear(cfg.value_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.embed_bn(self.embed(x)))
        t = h.flatten(2).transpose(1, 2) + self.pos_emb
        for block in self.blocks:
            t = block(t)
        t = self.final_ln(t)
        policy = self.policy_head(t).reshape(t.size(0), NUM_MOVES)
        v = t.mean(dim=1)
        value = torch.tanh(self.value_fc2(F.relu(self.value_fc1(v))))
        return policy, value


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
