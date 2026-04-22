"""
浅 CNN 位点编码器 (DeepBind / DeepSEA 风格).

输入: (N, 501, 4) one-hot 序列
输出: (N, output_dim) 位点表征

设计原则:
  - 3 层卷积 + 2 种 pooling 拼接
  - 总参数 ~300K, 比 DNABERT-2 小 60x
  - 局部 motif 检测能力强 (m6A 周围 DRACH motif, reader binding motif)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNSiteEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int = 501,
        in_channels: int = 4,
        conv_channels: tuple = (64, 128, 256),
        kernel_sizes: tuple = (11, 7, 5),
        pool_sizes: tuple = (4, 4, 1),     # 最后一层不 pool
        dropout: float = 0.2,
        output_dim: int = 256,
    ):
        super().__init__()
        assert len(conv_channels) == len(kernel_sizes) == len(pool_sizes)

        # --- 卷积栈 ---
        layers = []
        in_ch = in_channels
        for out_ch, k, p in zip(conv_channels, kernel_sizes, pool_sizes):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            if p > 1:
                layers.append(nn.MaxPool1d(p))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        # --- pooling: GlobalMax + GlobalAvg 拼接 ---
        # 最大池化: 抓"是否有强 motif"
        # 平均池化: 抓"整体序列特性"
        # 两者互补
        feat_dim = conv_channels[-1] * 2

        # --- 投影到 output_dim ---
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self._output_dim = output_dim

    def forward(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor | None = None,  # 兼容接口, CNN 不用
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        sequences : (N, L, 4) float  one-hot 序列
        attention_mask : 忽略 (兼容 DNABERT-2 接口)

        Returns
        -------
        (N, output_dim) float
        """
        # (N, L, 4) → (N, 4, L)  Conv1d 要求 channel-first
        x = sequences.transpose(1, 2)

        # 卷积栈
        h = self.conv(x)  # (N, C_last, L_after_pool)

        # 全局池化
        max_pool = F.adaptive_max_pool1d(h, 1).squeeze(-1)  # (N, C_last)
        avg_pool = F.adaptive_avg_pool1d(h, 1).squeeze(-1)  # (N, C_last)
        feat = torch.cat([max_pool, avg_pool], dim=-1)      # (N, 2*C_last)

        # 投影
        out = self.proj(feat)  # (N, output_dim)
        return out

    @property
    def output_dim(self) -> int:
        return self._output_dim
