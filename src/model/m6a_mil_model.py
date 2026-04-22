import torch
import torch.nn as nn

from src.model.site_encoder import DNABERT2SiteEncoder
from src.model.cnn_site_encoder import CNNSiteEncoder
from src.model.mil_aggregator import TransMILAggregator
from src.model.prediction_head import RegressionHead


class GatedFusion(nn.Module):
    def __init__(self, seq_dim: int, scalar_dim: int, hidden_dim: int):
        super().__init__()
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.scalar_proj = nn.Sequential(
            nn.Linear(scalar_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, h_seq: torch.Tensor, h_scalar: torch.Tensor) -> torch.Tensor:
        s = self.seq_proj(h_seq)
        c = self.scalar_proj(h_scalar)
        gate = self.gate_net(s + c)
        return gate * s + (1 - gate) * c


class M6AMIL(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        enc_cfg = config.get("site_encoder", {})
        agg_cfg = config.get("aggregator", {})
        head_cfg = config.get("head", {})
        scalar_dim = config.get("scalar_dim", 13)
        hidden_dim = config.get("hidden_dim", 256)

        # --- 1. 位点编码器 (根据 type 选择) ---
        encoder_type = enc_cfg.get("type", "dnabert2")
        if encoder_type == "dnabert2":
            self.site_encoder = DNABERT2SiteEncoder(
                model_name=enc_cfg.get("model_name", "zhihan1996/DNABERT-2-117M"),
                pooling=enc_cfg.get("pooling", "mean"),
                freeze_layers=enc_cfg.get("freeze_layers", 8),
            )
        elif encoder_type == "cnn":
            self.site_encoder = CNNSiteEncoder(
                seq_len=enc_cfg.get("seq_len", 501),
                conv_channels=tuple(enc_cfg.get("conv_channels", [64, 128, 256])),
                kernel_sizes=tuple(enc_cfg.get("kernel_sizes", [11, 7, 5])),
                pool_sizes=tuple(enc_cfg.get("pool_sizes", [4, 4, 1])),
                dropout=enc_cfg.get("dropout", 0.2),
                output_dim=enc_cfg.get("output_dim", 256),
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.encoder_type = encoder_type
        seq_dim = self.site_encoder.output_dim

        # --- 2. 门控融合 ---
        self.fusion = GatedFusion(
            seq_dim=seq_dim,
            scalar_dim=scalar_dim,
            hidden_dim=hidden_dim,
        )

        # --- 3. TransMIL 聚合器 ---
        self.aggregator = TransMILAggregator(
            dim=hidden_dim,
            n_heads=agg_cfg.get("n_heads", 4),
            n_layers=agg_cfg.get("n_layers", 2),
            dropout=agg_cfg.get("dropout", 0.1),
        )

        # --- 4. 预测头 ---
        self.head = RegressionHead(
            in_dim=hidden_dim,
            hidden_dim=head_cfg.get("hidden_dim", 128),
            n_tasks=head_cfg.get("n_tasks", 1),
            dropout=head_cfg.get("dropout", 0.2),
        )

    def forward(
        self,
        scalars: torch.Tensor,
        site_mask: torch.Tensor,
        # DNABERT-2 输入
        input_ids: torch.Tensor | None = None,
        token_attn_mask: torch.Tensor | None = None,
        # CNN 输入
        sequences: torch.Tensor | None = None,
    ) -> dict:
        """
        Parameters
        ----------
        scalars :         (B, S, D) float32  标量特征
        site_mask :       (B, S) bool        位点级 mask

        DNABERT-2 模式 (encoder_type='dnabert2'):
          input_ids :       (B, S, L) int64
          token_attn_mask : (B, S, L) int64

        CNN 模式 (encoder_type='cnn'):
          sequences :       (B, S, seq_len, 4) float32  one-hot
        """
        if self.encoder_type == "dnabert2":
            B, S, L = input_ids.shape
            real_mask = site_mask.view(-1)
            flat_ids = input_ids.view(B * S, L)
            flat_token_mask = token_attn_mask.view(B * S, L)

            real_ids = flat_ids[real_mask]
            real_token_mask = flat_token_mask[real_mask]

            real_repr = self.site_encoder(real_ids, real_token_mask)

            seq_repr = torch.zeros(
                B * S, real_repr.size(-1),
                device=real_repr.device, dtype=real_repr.dtype,
            )
            seq_repr[real_mask] = real_repr
            seq_repr = seq_repr.view(B, S, -1)

        elif self.encoder_type == "cnn":
            B, S, L, C = sequences.shape
            real_mask = site_mask.view(-1)
            flat_seq = sequences.view(B * S, L, C)
            real_seq = flat_seq[real_mask]              # (N_real, L, 4)

            real_repr = self.site_encoder(real_seq)     # (N_real, D)

            seq_repr = torch.zeros(
                B * S, real_repr.size(-1),
                device=real_repr.device, dtype=real_repr.dtype,
            )
            seq_repr[real_mask] = real_repr
            seq_repr = seq_repr.view(B, S, -1)

        # --- 后续: fusion + MIL ---
        site_repr = self.fusion(seq_repr, scalars)
        bag_repr, attn_weights = self.aggregator(site_repr, site_mask)
        predictions = self.head(bag_repr)

        return {
            "predictions": predictions,
            "attention": attn_weights,
        }

    def get_parameter_groups(self, lr_encoder: float, lr_head: float) -> list[dict]:
        encoder_params = [p for p in self.site_encoder.parameters() if p.requires_grad]
        encoder_ids = set(id(p) for p in encoder_params)
        head_params = [p for p in self.parameters()
                       if p.requires_grad and id(p) not in encoder_ids]
        return [
            {"params": encoder_params, "lr": lr_encoder},
            {"params": head_params, "lr": lr_head},
        ]