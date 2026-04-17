import math
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 

class TransMILAggregator(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
 
        # --- 位点间 self-attention ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN (更稳定)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )
 
        # --- Attention pooling ---
        self.attn_pool = GatedAttentionPooling(dim)


    def forward(
        self,
        instances: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        instances : (B, S, dim)
        mask : (B, S) bool, True=真实位点
 
        Returns
        -------
        bag_repr : (B, dim)
        attention_weights : (B, S)
        """
        src_key_padding_mask = ~mask  # (B, S), True=padding
        # 位点间 self-attention
        h = self.transformer(
            instances,
            src_key_padding_mask=src_key_padding_mask,
        )  # (B, S, dim)

        bag_repr, attn_weights = self.attn_pool(h, mask)
        return bag_repr, attn_weights
    
class GatedAttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn_v = nn.Linear(dim, dim)
        self.attn_u = nn.Linear(dim, dim)
        self.attn_w = nn.Linear(dim, 1)

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h : (B, S, dim)
        mask : (B, S) bool
 
        Returns
        -------
        pooled : (B, dim)
        weights : (B, S)
        """
        # Gated attention score
        v = torch.tanh(self.attn_v(h))       # (B, S, dim)
        u = torch.sigmoid(self.attn_u(h))    # (B, S, dim)
        scores = self.attn_w(v * u).squeeze(-1)  # (B, S)
 
        # Mask: padding 位置设为 -inf
        scores = scores.masked_fill(~mask, float("-inf"))
 
        # Softmax → weights
        weights = F.softmax(scores, dim=-1)  # (B, S)
 
        # 处理全 padding 的边界情况 (softmax(-inf) = nan)
        weights = weights.nan_to_num(0.0)
 
        # 加权求和
        pooled = torch.bmm(
            weights.unsqueeze(1),  # (B, 1, S)
            h,                      # (B, S, dim)
        ).squeeze(1)                # (B, dim)
 
        return pooled, weights
 