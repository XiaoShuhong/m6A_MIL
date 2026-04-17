import torch.nn as nn
 
 
class RegressionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        n_tasks: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_tasks),
        )
    def forward(self, x):
        """x: (B, in_dim) → (B, n_tasks)"""
        return self.head(x)
