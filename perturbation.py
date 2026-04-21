"""
单点消除扰动分析 (Single-site perturbation analysis).

对 test 集上的 family-exclusive m6A 位点 (YTHDF-only / IGF2BP-only) 做单点消除,
检验模型预测的变化方向是否符合生物学原理:
  - YTHDF (促降解):  删除后 KO 降解解除信号减弱 → log2FC 预测应 ↓ (Δ < 0)
  - IGF2BP (促稳定): 删除后 KO 稳定信号减弱     → log2FC 预测应 ↑ (Δ > 0)

三种消除方式:
  A (drop):      site_mask 置 False, 位点从 bag 中移除 (要求 bag_size >= 2)
  B (zero_rbp):  仅把 6 列 reader *_bound 置 0
  C (attenuate): B 的基础上, 额外把 m6a_level_wt / m6a_level_mettl3_dep 乘以 alpha
                 (alpha=0.0 → 完全失活; alpha=0.5 → 半失活)

用法:
    python perturbation.py --exp_dir experiments/dnabert2_transmil_baseline_fold1 \\
                           --fold 1 \\
                           --alpha 0.0
"""
from __future__ import annotations

import argparse
import json
import logging
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

from src.data.collate import m6a_collate_fn, InstanceBudgetSampler
from src.data.dataset import M6ADataset
from src.data.encoding import SequenceEncoder
from src.model.m6a_mil_model import M6AMIL
from src.utils.config import load_config

logger = logging.getLogger(__name__)


# =============================================================================
# 标量列在 scalars tensor 中的索引 (必须与 M6ADataset.SCALAR_COLS 保持一致)
# =============================================================================
# M6ADataset.SCALAR_COLS 顺序:
#   0: m6a_level_wt
#   1: m6a_level_mettl3_dep
#   2: ythdf1_bound
#   3: ythdf2_bound
#   4: ythdf3_bound
#   5: igf2bp1_bound
#   6: igf2bp2_bound
#   7: igf2bp3_bound
#   8: m6a_density_500nt
# 然后拼 4 维 region one-hot
IDX_M6A_LEVEL_WT = 0
IDX_M6A_LEVEL_MD = 1
IDX_YTHDF = [2, 3, 4]
IDX_IGF2BP = [5, 6, 7]
IDX_RBP_ALL = IDX_YTHDF + IDX_IGF2BP


# =============================================================================
# 工具: 分类每一行位点的 reader 家族
# =============================================================================
def classify_family(site_row: pd.Series) -> str:
    """返回 'ythdf_only' / 'igf2bp_only' / 'both' / 'none'."""
    ythdf = any(site_row.get(f"{r.lower()}_bound", False)
                for r in ["YTHDF1", "YTHDF2", "YTHDF3"])
    igf2bp = any(site_row.get(f"{r.lower()}_bound", False)
                 for r in ["IGF2BP1", "IGF2BP2", "IGF2BP3"])
    if ythdf and not igf2bp:
        return "ythdf_only"
    if igf2bp and not ythdf:
        return "igf2bp_only"
    if ythdf and igf2bp:
        return "both"
    return "none"


# =============================================================================
# 核心: 对一个 batch 做单点扰动
# =============================================================================
@torch.no_grad()
def perturb_batch(
    model: M6AMIL,
    batch: dict,
    device: torch.device,
    mode: str,
    alpha: float = 0.0,
) -> dict:
    """
    对 batch 内每个基因, 对其每个 family-exclusive 位点分别做单点扰动,
    返回 baseline 预测 + 每个扰动后的预测.

    Parameters
    ----------
    mode : 'drop' | 'zero_rbp' | 'attenuate'
    alpha : 仅 'attenuate' 使用. m6a_level 乘以 alpha (0.0 = 完全归零).

    Returns
    -------
    list[dict], 每个 dict 对应一个 (gene, site) 候选, 字段:
        gene_id, site_idx_in_bag, family, bag_size, pred_before, pred_after, delta
    """
    batch = _to_device(batch, device)

    # --- baseline: 原 batch 一次前向 ---
    base_out = model(
        input_ids=batch["input_ids"],
        token_attn_mask=batch["token_attn_mask"],
        scalars=batch["scalars"],
        site_mask=batch["site_mask"],
    )
    base_preds = base_out["predictions"].squeeze(-1).cpu().numpy()  # (B,)

    B = batch["input_ids"].shape[0]
    records = []

    # --- 对每个基因的每个候选位点构造扰动副本 ---
    # 做法: 把 "所有候选扰动" 拼成一个新 batch 一次前向, 而不是一个个单独跑
    # 一个候选 = 复制原基因的 (input_ids, token_attn_mask, scalars, site_mask),
    # 对指定位点做对应 mode 的修改
    pert_input_ids = []
    pert_token_mask = []
    pert_scalars = []
    pert_site_mask = []
    pert_meta = []  # list of (gene_batch_idx, site_idx, family)

    for b in range(B):
        gene_id = batch["gene_ids"][b]
        n_sites = batch["n_sites"][b]
        if n_sites == 0:
            continue

        # 该基因的所有位点的 scalars (仅真实位点范围)
        scalars_b = batch["scalars"][b, :n_sites].cpu().numpy()  # (n_sites, D)

        # 判断每个位点的家族
        ythdf_bound = scalars_b[:, IDX_YTHDF].max(axis=1) > 0.5   # (n_sites,)
        igf2bp_bound = scalars_b[:, IDX_IGF2BP].max(axis=1) > 0.5
        ythdf_only = ythdf_bound & (~igf2bp_bound)
        igf2bp_only = igf2bp_bound & (~ythdf_bound)

        # 候选位点索引
        candidates = []
        for i in range(n_sites):
            if ythdf_only[i]:
                candidates.append((i, "ythdf_only"))
            elif igf2bp_only[i]:
                candidates.append((i, "igf2bp_only"))

        if not candidates:
            continue

        # mode A (drop) 要求基因至少 2 个位点 (否则 bag 空了)
        if mode == "drop" and n_sites < 2:
            continue

        # 为该基因的每个候选位点生成一个扰动副本
        for site_idx, family in candidates:
            # 从原 batch 切出这一行, 深拷贝
            ids_row = batch["input_ids"][b].clone()           # (S, L)
            tok_row = batch["token_attn_mask"][b].clone()     # (S, L)
            sca_row = batch["scalars"][b].clone()             # (S, D)
            msk_row = batch["site_mask"][b].clone()           # (S,)

            if mode == "drop":
                msk_row[site_idx] = False
            elif mode == "zero_rbp":
                for j in IDX_RBP_ALL:
                    sca_row[site_idx, j] = 0.0
            elif mode == "attenuate":
                for j in IDX_RBP_ALL:
                    sca_row[site_idx, j] = 0.0
                sca_row[site_idx, IDX_M6A_LEVEL_WT] *= alpha
                sca_row[site_idx, IDX_M6A_LEVEL_MD] *= alpha
            else:
                raise ValueError(f"Unknown mode: {mode}")

            pert_input_ids.append(ids_row)
            pert_token_mask.append(tok_row)
            pert_scalars.append(sca_row)
            pert_site_mask.append(msk_row)
            pert_meta.append({
                "gene_batch_idx": b,
                "gene_id": int(gene_id),
                "site_idx": int(site_idx),
                "family": family,
                "bag_size": int(n_sites),
                "pred_before": float(base_preds[b]),
            })

    if not pert_meta:
        return []

    # --- 把所有扰动副本拼成一个 batch, 一次前向 ---
    # 但这个拼出来的 batch 可能很大 (一个 batch 内可能有几十个候选位点),
    # 分 chunk 跑
    chunk = 16
    all_pert_preds = []
    for start in range(0, len(pert_meta), chunk):
        end = min(start + chunk, len(pert_meta))
        ids_chunk = torch.stack(pert_input_ids[start:end], dim=0)
        tok_chunk = torch.stack(pert_token_mask[start:end], dim=0)
        sca_chunk = torch.stack(pert_scalars[start:end], dim=0)
        msk_chunk = torch.stack(pert_site_mask[start:end], dim=0)

        out = model(
            input_ids=ids_chunk,
            token_attn_mask=tok_chunk,
            scalars=sca_chunk,
            site_mask=msk_chunk,
        )
        all_pert_preds.append(out["predictions"].squeeze(-1).cpu().numpy())

    pert_preds = np.concatenate(all_pert_preds)

    for meta, pa in zip(pert_meta, pert_preds):
        records.append({
            "gene_id": meta["gene_id"],
            "site_idx": meta["site_idx"],
            "family": meta["family"],
            "bag_size": meta["bag_size"],
            "pred_before": meta["pred_before"],
            "pred_after": float(pa),
            "delta": float(pa) - meta["pred_before"],
        })

    return records


def _to_device(batch: dict, device: torch.device) -> dict:
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device, non_blocking=True)
        else:
            result[k] = v
    return result


# =============================================================================
# 统计分析
# =============================================================================
def analyze(records: list[dict], mode: str) -> dict:
    """对一种模式的结果做分析."""
    if not records:
        return {"mode": mode, "n": 0}

    df = pd.DataFrame(records)
    summary = {"mode": mode, "n_total": len(df)}

    for family, expected_sign in [("ythdf_only", -1), ("igf2bp_only", +1)]:
        sub = df[df["family"] == family]
        if len(sub) == 0:
            continue
        deltas = sub["delta"].values
        # 方向一致性
        consistent = (np.sign(deltas) == expected_sign).sum()
        # Wilcoxon signed-rank (H0: median(delta) == 0)
        if len(deltas) >= 5:
            try:
                _, p_wilcox = stats.wilcoxon(deltas, alternative=(
                    "less" if expected_sign < 0 else "greater"
                ))
            except ValueError:
                p_wilcox = np.nan
        else:
            p_wilcox = np.nan

        summary[family] = {
            "n": len(sub),
            "expected_sign": expected_sign,
            "delta_mean": float(deltas.mean()),
            "delta_median": float(np.median(deltas)),
            "delta_std": float(deltas.std()),
            "pct_correct_direction": float(consistent / len(deltas)),
            "p_wilcoxon_onesided": float(p_wilcox) if not np.isnan(p_wilcox) else None,
        }

    return summary


def print_summary(summaries: list[dict]) -> None:
    logger.info("=" * 70)
    logger.info("PERTURBATION RESULTS")
    logger.info("=" * 70)
    for s in summaries:
        logger.info("")
        logger.info("Mode: %s  (total candidates: %d)", s["mode"], s.get("n_total", 0))
        logger.info("-" * 70)
        for family in ["ythdf_only", "igf2bp_only"]:
            if family not in s:
                continue
            f = s[family]
            expected = "↓ (Δ<0)" if f["expected_sign"] < 0 else "↑ (Δ>0)"
            p_str = f"{f['p_wilcoxon_onesided']:.2e}" if f["p_wilcoxon_onesided"] is not None else "N/A"
            logger.info(
                "  %-12s  n=%4d  expected=%s  "
                "Δ mean=%+.4f  median=%+.4f  std=%.4f  "
                "correct_dir=%.1f%%  p=%s",
                family, f["n"], expected,
                f["delta_mean"], f["delta_median"], f["delta_std"],
                100 * f["pct_correct_direction"], p_str,
            )
    logger.info("=" * 70)


# =============================================================================
# 主流程
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="实验目录, 含 config.yaml 和 checkpoints/best.pt")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="attenuate 模式的 m6a_level 缩放系数")
    parser.add_argument("--modes", type=str, default="drop,zero_rbp,attenuate",
                        help="逗号分隔的模式列表")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "val", "train"],
                        help="在哪个 split 上做扰动")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    exp_dir = Path(args.exp_dir)
    config = load_config(exp_dir / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- 加载数据表 ---
    gene_table = pd.read_parquet(config["data"]["gene_table"])
    if gene_table.index.name == "gene_id_entrez":
        gene_table = gene_table.reset_index()
    site_table = pd.read_parquet(config["data"]["site_table"])

    # --- 加载 split ---
    split_path = Path(config["data"]["split_dir"]) / f"fold_{args.fold}.json"
    with open(split_path) as f:
        split = json.load(f)
    gene_ids = split[f"{args.split}_gene_ids"]
    logger.info("Using %s split: %d genes", args.split, len(gene_ids))

    # --- 过滤 baseMean >= 10 (标签可信, 和训练一致) ---
    gt_idx = gene_table.set_index("gene_id_entrez")
    valid = [g for g in gene_ids
             if g in gt_idx.index
             and pd.notna(gt_idx.loc[g, "basemean"])
             and gt_idx.loc[g, "basemean"] >= config["data"].get("min_basemean", 10)
             and pd.notna(gt_idx.loc[g, "log2fc"])]
    logger.info("After baseMean filter: %d genes", len(valid))

    # --- 构建 Dataset / DataLoader ---
    encoder = SequenceEncoder(
        method=config["model"]["site_encoder"]["type"],
        seq_len=config["data"]["seq_len"],
    )
    collate = partial(m6a_collate_fn, encoder=encoder)
    ds = M6ADataset(
        gene_ids=valid,
        gene_table=gene_table,
        site_table=site_table,
        seq_dir=config["data"]["seq_dir"],
        config=config["data"],
    )
    sampler = InstanceBudgetSampler(
        n_sites_per_gene=ds.n_sites_per_gene,
        instance_budget=config["training"]["instance_budget"],
        max_batch_size=config["training"]["max_batch_size"],
        min_batch_size=1,
        shuffle=False,
    )
    loader = DataLoader(
        ds, batch_sampler=sampler, collate_fn=collate,
        num_workers=0, pin_memory=True,
    )

    # --- 加载模型 ---
    model = M6AMIL(config["model"]).to(device)
    ckpt = exp_dir / "checkpoints" / "best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    logger.info("Loaded checkpoint: %s", ckpt)

    # --- 对每种 mode 跑扰动 ---
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    all_summaries = []
    out_dir = exp_dir / "results" / f"perturbation_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        logger.info("-" * 70)
        logger.info("Running mode: %s", mode + (f" (alpha={args.alpha})"
                                                if mode == "attenuate" else ""))
        all_records = []
        for batch in loader:
            recs = perturb_batch(model, batch, device, mode=mode, alpha=args.alpha)
            all_records.extend(recs)
        logger.info("  Collected %d (gene, site) candidates", len(all_records))

        # 保存
        df = pd.DataFrame(all_records)
        suffix = f"_alpha{args.alpha}" if mode == "attenuate" else ""
        df.to_parquet(out_dir / f"{mode}{suffix}.parquet", index=False)

        # 分析
        summary = analyze(all_records, mode + suffix)
        all_summaries.append(summary)

    # --- 打印汇总 ---
    print_summary(all_summaries)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    logger.info("All results saved to %s", out_dir)


if __name__ == "__main__":
    main()
