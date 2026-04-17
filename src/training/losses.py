import torch
import torch.nn as nn
 

class WeightedMSELoss(nn.Module):
    def __init__(self, use_weights: bool = True):
        super().__init__()
        self.use_weights = use_weights

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:

        predictions = predictions.squeeze(-1)  # (B,)
 
        residuals = (predictions - targets) ** 2  # (B,)
 
        if self.use_weights and sample_weights is not None:
            # 归一化权重: 均值为 1, 不改变 loss 的量级
            w = sample_weights / sample_weights.mean().clamp(min=1e-8)
            loss = (w * residuals).mean()
        else:
            loss = residuals.mean()
 
        return loss

def compute_sample_weights(basemean: torch.Tensor) -> torch.Tensor:
    """
    从 baseMean 计算样本权重.
 
    w = log2(baseMean + 1)
    然后归一化到 [0, 1] 范围.
 
    Parameters
    ----------
    basemean : (B,) float tensor
 
    Returns
    -------
    (B,) float tensor, 权重
    """
    w = torch.log2(basemean + 1)
    w = w / w.max().clamp(min=1e-8)
    return w