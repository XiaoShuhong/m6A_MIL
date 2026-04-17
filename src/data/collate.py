from __future__ import annotations
 
import random
 
import numpy as np
import torch
from torch.utils.data import Sampler
 
from src.data.encoding import SequenceEncoder


def m6a_collate_fn(
    batch: list[dict],
    encoder: SequenceEncoder,
) -> dict:
    B = len(batch)
    max_sites = max(item["n_sites"] for item in batch)
    if max_sites == 0:
        max_sites = 1

    scalar_dim = 0
    for item in batch:
        if item["n_sites"] > 0:
            scalar_dim = item["scalars"].shape[1]
            break

    if encoder.output_type == "dnabert2":
        result = _collate_dnabert2(batch, encoder, B, max_sites, scalar_dim)
    elif encoder.output_type == "onehot":
        result = _collate_onehot(batch, encoder, B, max_sites, scalar_dim)
    else:
        raise ValueError(f"Unknown encoder type: {encoder.output_type}")
 
    return result
 
def _collate_dnabert2(
    batch: list[dict],
    encoder: SequenceEncoder,
    B: int,
    max_sites: int,
    scalar_dim: int,
) -> dict:
    
    max_tokens = encoder.max_tokens
 
    all_input_ids = np.zeros((B, max_sites, max_tokens), dtype=np.int64)
    all_token_masks = np.zeros((B, max_sites, max_tokens), dtype=np.int64)
    all_scalars = np.zeros((B, max_sites, scalar_dim), dtype=np.float32)
    all_site_masks = np.zeros((B, max_sites), dtype=bool)
    all_labels = []
    all_gene_ids = []
    all_n_sites = []
    all_basemeans = []

    for i, item in enumerate(batch):
        n = item["n_sites"]
        all_labels.append(item["label"])
        all_gene_ids.append(item["gene_id"])
        all_n_sites.append(n)
        all_basemeans.append(item["basemean"])
 
        if n == 0:
            continue
 
        # 编码序列
        encoded = encoder.encode_batch(item["sequences"])  # dict
        all_input_ids[i, :n, :] = encoded["input_ids"]         # (n, L)
        all_token_masks[i, :n, :] = encoded["attention_mask"]   # (n, L)

        # 标量
        all_scalars[i, :n, :] = item["scalars"]
 
        # 位点 mask
        all_site_masks[i, :n] = True

    return {
        "input_ids": torch.from_numpy(all_input_ids),                  # (B, S, L)
        "token_attn_mask": torch.from_numpy(all_token_masks),          # (B, S, L)
        "scalars": torch.from_numpy(all_scalars).float(),              # (B, S, D)
        "site_mask": torch.from_numpy(all_site_masks),                 # (B, S)
        "labels": torch.tensor(all_labels, dtype=torch.float32),       # (B,)
        "basemean": torch.tensor(all_basemeans, dtype=torch.float32),
        "gene_ids": all_gene_ids,
        "n_sites": all_n_sites,
    }
    
def _collate_onehot(
    batch: list[dict],
    encoder: SequenceEncoder,
    B: int,
    max_sites: int,
    scalar_dim: int,
) -> dict:
    seq_len = encoder.seq_len
 
    all_sequences = np.zeros((B, max_sites, seq_len, 4), dtype=np.float32)
    all_scalars = np.zeros((B, max_sites, scalar_dim), dtype=np.float32)
    all_site_masks = np.zeros((B, max_sites), dtype=bool)
    all_labels = []
    all_gene_ids = []
    all_n_sites = []

    for i, item in enumerate(batch):
        n = item["n_sites"]
        all_labels.append(item["label"])
        all_gene_ids.append(item["gene_id"])
        all_n_sites.append(n)
 
        if n == 0:
            continue

        encoded = encoder.encode_batch(item["sequences"])  # (n, seq_len, 4)
        all_sequences[i, :n, :, :] = encoded
        all_scalars[i, :n, :] = item["scalars"]
        all_site_masks[i, :n] = True
 
    return {
        "sequences": torch.from_numpy(all_sequences),                  # (B, S, L, 4)
        "scalars": torch.from_numpy(all_scalars).float(),              # (B, S, D)
        "site_mask": torch.from_numpy(all_site_masks),                 # (B, S)
        "labels": torch.tensor(all_labels, dtype=torch.float32),       # (B,)
        "gene_ids": all_gene_ids,
        "n_sites": all_n_sites,
    }


class InstanceBudgetSampler(Sampler):
    def __init__(
        self,
        n_sites_per_gene: np.ndarray,
        instance_budget: int = 512,
        max_batch_size: int = 64,
        min_batch_size: int = 4,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        
        self.n_sites = n_sites_per_gene.copy()
        self.instance_budget = instance_budget
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0
 
        self._batches = self._build_batches()
    
    def _build_batches(self) -> list[list[int]]:
    
        sorted_indices = np.argsort(self.n_sites).tolist()
 
        batches = []
        current_batch = []
        current_instances = 0
 
        for idx in sorted_indices:
            n = max(int(self.n_sites[idx]), 1)
 
            if (current_instances + n > self.instance_budget
                    or len(current_batch) >= self.max_batch_size):
                if len(current_batch) >= self.min_batch_size:
                    batches.append(current_batch)
                elif batches:
                    batches[-1].extend(current_batch)
                current_batch = [idx]
                current_instances = n
            else:
                current_batch.append(idx)
                current_instances += n
 
        if current_batch:
            if self.drop_last and len(current_batch) < self.min_batch_size:
                pass
            else:
                batches.append(current_batch)
 
        return batches

    def __iter__(self):
        batches = self._batches.copy()
        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(batches)
            self._epoch += 1
        for batch in batches:
            yield batch
    
    def __len__(self) -> int:
        return len(self._batches)
    
    def batch_stats(self) -> dict:
        """打印 batch 统计信息."""
        sizes = [len(b) for b in self._batches]
        instances = [
            sum(max(int(self.n_sites[i]), 1) for i in b)
            for b in self._batches
        ]
        max_sites = [max(int(self.n_sites[i]) for i in b) for b in self._batches]
        return {
            "n_batches": len(self._batches),
            "genes_per_batch": f"mean={np.mean(sizes):.1f}, min={min(sizes)}, max={max(sizes)}",
            "instances_per_batch": f"mean={np.mean(instances):.1f}, min={min(instances)}, max={max(instances)}",
            "max_sites_in_batch": f"mean={np.mean(max_sites):.1f}, max={max(max_sites)}",
        }