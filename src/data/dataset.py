from __future__ import annotations
 
import logging
from pathlib import Path
from typing import Optional
 
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
 
logger = logging.getLogger(__name__)
 
class M6ADataset(Dataset):

    SCALAR_COLS = [
        "m6a_level_wt",
        "m6a_level_mettl3_dep",
        "ythdf1_bound",
        "ythdf2_bound",
        "ythdf3_bound",
        "igf2bp1_bound",
        "igf2bp2_bound",
        "igf2bp3_bound",
        "m6a_density_500nt",
    ]
    REGION_MAP = {"5UTR": 0, "CDS": 1, "3UTR": 2, "intron": 3}
    def __init__(
        self,
        gene_ids: list[int],
        gene_table: pd.DataFrame,
        site_table: pd.DataFrame,
        seq_dir: str | Path,
        config: dict,
    ):
        self.seq_dir = Path(seq_dir)
        self.max_sites = config.get("max_sites_per_gene", 50)
        # --- 构建基因索引 ---

        gt = gene_table.copy()
        if gt.index.name != "gene_id_entrez":
            gt = gt.set_index("gene_id_entrez")
        # 只保留当前 split 的基因
        self.gene_ids = [gid for gid in gene_ids if gid in gt.index]
        self.gene_table = gt.loc[self.gene_ids]

        # --- 构建位点分组索引 ---
        st = site_table[site_table["gene_id_entrez"].isin(self.gene_ids)].copy()
        # 构造 site_id
        st["site_id"] = (
            st["chrom"] + "_" + st["pos"].astype(str) + "_" + st["strand"]
        )
        self.site_groups = {
            gid: group for gid, group in st.groupby("gene_id_entrez")
        }

        # --- 预计算每个基因的位点数 (用于 BucketBatchSampler) ---
        self.n_sites_per_gene = np.array([
            len(self.site_groups.get(gid, [])) for gid in self.gene_ids
        ], dtype=np.int32)
        # --- HDF5 文件句柄缓存 (懒加载) ---
        self._h5_cache: dict[str, h5py.File] = {}
        # --- 日志 ---
        n_with_sites = int((self.n_sites_per_gene > 0).sum())
        logger.info(
            "M6ADataset: %d genes, %d with m6A sites, "
            "sites/gene: median=%d, mean=%.1f, max=%d",
            len(self.gene_ids),
            n_with_sites,
            int(np.median(self.n_sites_per_gene[self.n_sites_per_gene > 0]))
            if n_with_sites > 0 else 0,
            self.n_sites_per_gene.mean(),
            int(self.n_sites_per_gene.max()),
        )
    def __len__(self) -> int:
        return len(self.gene_ids)
    
    def __getitem__(self, idx: int) -> dict:
        gene_id = self.gene_ids[idx]
        gene_row = self.gene_table.loc[gene_id]
        basemean = float(gene_row["basemean"])
        # --- 标签 ---
        label = float(gene_row["log2fc"])

        # --- 位点数据 ---
        if gene_id not in self.site_groups:
            # 空 bag (无 m6A 的基因, 理论上不应出现在主数据集)
            return {
                "gene_id": gene_id,
                "label": label,
                "sequences": [],
                "scalars": np.empty((0, self._scalar_dim()), dtype=np.float32),
                "n_sites": 0,
            }
        
        sites = self.site_groups[gene_id]
        if len(sites) > self.max_sites:
            sites = sites.sample(n=self.max_sites, random_state=gene_id % 10000)
        
        # --- 提取序列 ---
        sequences = []
        for _, site in sites.iterrows():
            seq = self._read_sequence(site["chrom"], site["site_id"])
            sequences.append(seq)
        
        # --- 提取标量特征 ---
        scalars = self._extract_scalars(sites)
 
        return {
            "gene_id": gene_id,
            "label": label,
            "sequences": sequences,       # list[str], 每个 501 chars
            "scalars": scalars,            # np.ndarray, (n_sites, scalar_dim)
            "basemean": basemean,
            "n_sites": len(sequences),
        }
    
    def _read_sequence(self, chrom: str, site_id: str) -> str:
        if chrom not in self._h5_cache:
            h5_path = self.seq_dir / f"{chrom}.h5"
            self._h5_cache[chrom] = h5py.File(str(h5_path), "r")
 
        h5 = self._h5_cache[chrom]
        raw = h5[site_id][()]
        if isinstance(raw, bytes):
            return raw.decode()
        return str(raw)
    
    def _extract_scalars(self, sites: pd.DataFrame) -> np.ndarray:
        
        parts = []
 
        # 数值特征
        for col in self.SCALAR_COLS:
            if col in sites.columns:
                vals = sites[col].values.astype(np.float32)
            else:
                vals = np.zeros(len(sites), dtype=np.float32)
            parts.append(vals.reshape(-1, 1))
 
        # 区域 one-hot (4 维: 5UTR, CDS, 3UTR, intron)
        if "region" in sites.columns:
            region_onehot = np.zeros((len(sites), len(self.REGION_MAP)), dtype=np.float32)
            for i, region in enumerate(sites["region"].values):
                idx = self.REGION_MAP.get(region, -1)
                if idx >= 0:
                    region_onehot[i, idx] = 1.0
            parts.append(region_onehot)
 
        return np.hstack(parts)  # (n_sites, scalar_dim)
    
    def _scalar_dim(self) -> int:

        return len(self.SCALAR_COLS) + len(self.REGION_MAP)

    @property
    def scalar_dim(self) -> int:
        return self._scalar_dim()
 
    def close(self):
        for h5 in self._h5_cache.values():
            h5.close()
        self._h5_cache.clear()
 
    def __del__(self):
        self.close()