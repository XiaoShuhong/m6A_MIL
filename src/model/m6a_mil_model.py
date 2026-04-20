import torch
import torch.nn as nn
 
from src.model.site_encoder import DNABERT2SiteEncoder
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
        s = self.seq_proj(h_seq)       # (B, S, hidden_dim)
        c = self.scalar_proj(h_scalar) # (B, S, hidden_dim)
        gate = self.gate_net(s + c)    # (B, S, hidden_dim)
        return gate * s + (1 - gate) * c

class M6AMIL(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
 
        # --- 配置 ---
        enc_cfg = config.get("site_encoder", {})
        agg_cfg = config.get("aggregator", {})
        head_cfg = config.get("head", {})
        scalar_dim = config.get("scalar_dim", 13)
        hidden_dim = config.get("hidden_dim", 256)

        # --- 1. DNABERT-2 位点编码器 ---
        self.site_encoder = DNABERT2SiteEncoder(
            model_name=enc_cfg.get("model_name", "zhihan1996/DNABERT-2-117M"),
            pooling=enc_cfg.get("pooling", "mean"),
            freeze_layers=enc_cfg.get("freeze_layers", 8),
        )
        seq_dim = self.site_encoder.output_dim  # 768
 
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
        input_ids: torch.Tensor,
        token_attn_mask: torch.Tensor,
        scalars: torch.Tensor,
        site_mask: torch.Tensor,
    ) -> dict:
        """
        Parameters
        ----------
        input_ids :       (B, S, L) int64    DNABERT-2 token ids
        token_attn_mask : (B, S, L) int64    token 级 attention mask
        scalars :         (B, S, D) float32  标量特征
        site_mask :       (B, S) bool        位点级 mask
 
        Returns
        -------
        dict:
            predictions: (B, 1) float   log2FC 预测值
            attention:   (B, S) float   每个位点的 attention 权重
        """
        B, S, L = input_ids.shape
 
        # --- Step 1: DNABERT-2 编码 ---
        real_mask = site_mask.view(-1)                         # (B*S,)
        flat_ids = input_ids.view(B * S, L)                    # (B*S, L)
        flat_token_mask = token_attn_mask.view(B * S, L)       # (B*S, L)

        real_ids = flat_ids[real_mask]                          # (N_real, L)
        real_token_mask = flat_token_mask[real_mask]            # (N_real, L)
        # print(f"DEBUG: B={B}, S={S}, L={L}, B*S={B*S}, real={real_ids.shape[0]}, "
        #     f"real_ids.shape={real_ids.shape}, dtype={real_ids.dtype}, "
        #     f"max_id={real_ids.max().item()}")

        real_repr = self.site_encoder(real_ids, real_token_mask) 


        seq_repr = torch.zeros(B * S, real_repr.size(-1),
                           device=real_repr.device, dtype=real_repr.dtype)
        seq_repr[real_mask] = real_repr
        seq_repr = seq_repr.view(B, S, -1)                     # (B, S, 768)

        # --- Step 2: 门控融合 ---
        site_repr = self.fusion(seq_repr, scalars)  # (B, S, hidden_dim)

        # --- Step 3: TransMIL 聚合 ---
        bag_repr, attn_weights = self.aggregator(
            site_repr, site_mask,
        )  # bag_repr: (B, hidden_dim), attn_weights: (B, S)
 
        # --- Step 4: 预测 ---
        predictions = self.head(bag_repr)  # (B, 1)
 
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