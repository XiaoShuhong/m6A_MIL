import torch
import torch.nn as nn
from transformers import AutoConfig,AutoModel
 

class DNABERT2SiteEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "zhihan1996/DNABERT-2-117M",
        pooling: str = "mean",
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True,
        )
        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = 0
       

        self.bert = AutoModel.from_pretrained(
            model_name, config=config, trust_remote_code=True, low_cpu_mem_usage=False,   
        )
        self.pooling = pooling
        self.hidden_dim = self.bert.config.hidden_size  # 768
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : (N, L) int64
        attention_mask : (N, L) int64
 
        Returns
        -------
        (N, hidden_dim) float tensor
        """
        N = input_ids.size(0)
        chunk_size = 16
        if N <= chunk_size:
            return self._encode(input_ids, attention_mask)
        all_repr = []
        for i in range(0, N, chunk_size):
            repr = self._encode(
                input_ids[i:i+chunk_size],
                attention_mask[i:i+chunk_size],
            )
            all_repr.append(repr)
        return torch.cat(all_repr, dim=0)  # (N, 768)
    

    def _encode(self, input_ids, attention_mask):
        """编码一个 chunk."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden = outputs[0]  # (chunk, L, 768)
        if self.pooling == "cls":
            return last_hidden[:, 0, :]
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1)

    @property
    def output_dim(self) -> int:
        return self.hidden_dim
    