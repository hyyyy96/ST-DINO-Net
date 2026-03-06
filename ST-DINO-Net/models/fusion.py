import torch
import torch.nn as nn
import torch.nn.functional as F


class BiDirectionalGatedFusion(nn.Module):
    """
    Bidirectional Gated Fusion Module (BDGF)
    Fuses spatial and motion features through cross-attention and adaptive gating
    """

    def __init__(self, dim_s, dim_m, num_heads=8, dropout=0.1):
        super().__init__()
        self.common_dim = dim_s
        self.proj_m = nn.Linear(dim_m, self.common_dim)
        self.norm_m = nn.LayerNorm(self.common_dim)
        self.norm_s = nn.LayerNorm(dim_s)

        self.attn_s2m = nn.MultiheadAttention(
            embed_dim=self.common_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_m2s = nn.MultiheadAttention(
            embed_dim=self.common_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.gate_net = nn.Sequential(
            nn.Linear(2 * self.common_dim, self.common_dim // 2),
            nn.ReLU(),
            nn.Linear(self.common_dim // 2, 1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(self.common_dim)

    def forward(self, x_s, x_m):
        # Normalize inputs
        x_s = self.norm_s(x_s)
        x_m_proj = self.norm_m(self.proj_m(x_m))

        # Add sequence dimension for attention
        q_s = x_s.unsqueeze(1)
        q_m = x_m_proj.unsqueeze(1)

        # Spatial → Motion attention
        out_s2m, _ = self.attn_s2m(query=q_s, key=q_m, value=q_m)
        out_s2m = out_s2m.squeeze(1) + x_s

        # Motion → Spatial attention
        out_m2s, _ = self.attn_m2s(query=q_m, key=q_s, value=q_s)
        out_m2s = out_m2s.squeeze(1) + x_m_proj

        # Adaptive gating
        combined = torch.cat([out_s2m, out_m2s], dim=-1)
        alpha = self.gate_net(combined)

        # Fuse
        fused = alpha * out_s2m + (1 - alpha) * out_m2s
        fused = self.norm_out(fused)
        fused = self.dropout(fused)

        return fused