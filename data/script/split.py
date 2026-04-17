
from __future__ import annotations
 
import json
import logging
from pathlib import Path
 
import pandas as pd
 
logger = logging.getLogger(__name__)

def _chrom_sort_key(chrom: str) -> int:
    c = chrom.replace("chr", "")
    if c == "X":
        return 23
    elif c == "Y":
        return 24
    elif c == "M":
        return 25
    else:
        return int(c)
    
def _fmt_chroms(chroms) -> str:
    return ", ".join(sorted(chroms, key=_chrom_sort_key))

def greedy_balanced_partition(chrom_gene_counts: dict, n_groups: int = 5) -> list[list[str]]:
    """
    贪心平衡分组: 按基因数从大到小, 每次分配到当前最小的组.
    """
    sorted_chroms = sorted(chrom_gene_counts.items(), key=lambda x: -x[1])
    groups: list[list[str]] = [[] for _ in range(n_groups)]
    group_sizes: list[int] = [0] * n_groups
 
    for chrom, count in sorted_chroms:
        min_idx = group_sizes.index(min(group_sizes))
        groups[min_idx].append(chrom)
        group_sizes[min_idx] += count
 
    for i, (grp, size) in enumerate(zip(groups, group_sizes)):
        logger.info("  Group %d: %4d genes  [%s]", i + 1, size, _fmt_chroms(grp))
 
    return groups

def pick_val_chroms(
    group_chroms: list[str],
    chrom_gene_counts: dict,
    target_val_pct: float,
    total_genes: int,
) -> list[str]:
    """
    从一个染色体组中选出 1-2 条最小的染色体作为 val.
    目标: val 占总基因数的 target_val_pct 左右.
    """
    sorted_by_size = sorted(group_chroms, key=lambda c: chrom_gene_counts.get(c, 0))
    target_n = int(total_genes * target_val_pct)
 
    val_chroms = []
    val_n = 0
    for chrom in sorted_by_size:
        n = chrom_gene_counts.get(chrom, 0)
        if val_n + n <= target_n * 1.5:
            val_chroms.append(chrom)
            val_n += n
            if val_n >= target_n * 0.7:
                break
        else:
            if not val_chroms:
                val_chroms.append(chrom)
                val_n += n
            break
 
    return val_chroms
 


def make_splits(
    gene_table_path: str | Path,
    site_table_path: str | Path,
    output_dir: str | Path,
    n_folds: int = 5,
    target_val_pct: float = 0.07,
    min_basemean: float = 0.0,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    # =========================================================================
    # 1. 读取数据
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Loading data...")
 
    gt = pd.read_parquet(gene_table_path)
    if gt.index.name == "gene_id_entrez":
        gt = gt.reset_index()
 
    st = pd.read_parquet(site_table_path)
    logger.info("  gene_table: %d genes", len(gt))
    logger.info("  site_table: %d rows", len(st))
     # =========================================================================
    # 2. 标记有/无 m6A
    # =========================================================================
    genes_with_m6a = set(st["gene_id_entrez"].unique())
    gt["has_m6a"] = gt["gene_id_entrez"].isin(genes_with_m6a)
 
    logger.info("  有 m6A 位点的基因: %d (%.1f%%)",
                int(gt["has_m6a"].sum()), 100 * gt["has_m6a"].mean())
    logger.info("  无 m6A 位点的基因: %d", int((~gt["has_m6a"]).sum()))

    # =========================================================================
    # 3. 基本过滤
    # =========================================================================
    has_log2fc = gt["log2fc"].notna()
    has_chrom = gt["chrom"].notna()
    has_expression = gt["basemean"] >= min_basemean if min_basemean > 0 else pd.Series(True, index=gt.index)
 
    valid = has_log2fc & has_chrom & has_expression
    gt_valid = gt[valid].copy()
    logger.info("  通过过滤的基因: %d", len(gt_valid))
 
    gt_m6a = gt_valid[gt_valid["has_m6a"]].copy()
    gt_neg = gt_valid[~gt_valid["has_m6a"]].copy()
    logger.info("  主数据集 (有 m6A): %d", len(gt_m6a))
    logger.info("  阴性对照 (无 m6A): %d", len(gt_neg))

    # =========================================================================
    # 4. 统计 + 贪心分组
    # =========================================================================
    chrom_counts = gt_m6a["chrom"].value_counts().to_dict()
    total_m6a_genes = len(gt_m6a)
 
    logger.info("-" * 60)
    logger.info("各染色体基因数 (有 m6A):")
    for chrom in sorted(chrom_counts.keys(), key=_chrom_sort_key):
        logger.info("  %-5s: %4d (%.1f%%)", chrom, chrom_counts[chrom],
                     100 * chrom_counts[chrom] / total_m6a_genes)
 
    logger.info("-" * 60)
    logger.info("贪心平衡分组 (%d groups):", n_folds)
    groups = greedy_balanced_partition(chrom_counts, n_groups=n_folds)
    # =========================================================================
    # 5. 生成 5 折
    # =========================================================================
    logger.info("=" * 60)
    logger.info("生成 %d-fold 划分 (target val ≈ %.0f%%):", n_folds, target_val_pct * 100)
 
    all_folds = {}
    summary_stats = {
        "n_folds": n_folds,
        "target_val_pct": target_val_pct,
        "min_basemean": min_basemean,
        "total_genes_with_m6a": len(gt_m6a),
        "total_genes_neg_control": len(gt_neg),
        "folds": {},
    }
 
    for fold_idx in range(n_folds):
        # test = 当前组的全部染色体
        test_group_idx = fold_idx
        test_chroms = set(groups[test_group_idx])
 
        # val = 下一组中选 1-2 条最小的染色体
        val_source_idx = (fold_idx + 1) % n_folds
        val_chroms_list = pick_val_chroms(
            groups[val_source_idx], chrom_counts, target_val_pct, total_m6a_genes,
        )
        val_chroms = set(val_chroms_list)
 
        # train = 剩余所有
        train_chroms = set()
        for i in range(n_folds):
            if i == test_group_idx:
                continue
            for c in groups[i]:
                if c not in val_chroms:
                    train_chroms.add(c)
 
        # 分配基因
        test_genes = gt_m6a[gt_m6a["chrom"].isin(test_chroms)]["gene_id_entrez"].tolist()
        val_genes = gt_m6a[gt_m6a["chrom"].isin(val_chroms)]["gene_id_entrez"].tolist()
        train_genes = gt_m6a[gt_m6a["chrom"].isin(train_chroms)]["gene_id_entrez"].tolist()
        total = len(test_genes) + len(val_genes) + len(train_genes)
 
        # 位点数
        n_test_sites = int(st["gene_id_entrez"].isin(test_genes).sum())
        n_val_sites = int(st["gene_id_entrez"].isin(val_genes).sum())
        n_train_sites = int(st["gene_id_entrez"].isin(train_genes).sum())
 
        # log2FC 分布
        train_lfc = gt_m6a[gt_m6a["gene_id_entrez"].isin(train_genes)]["log2fc"]
        val_lfc = gt_m6a[gt_m6a["gene_id_entrez"].isin(val_genes)]["log2fc"]
        test_lfc = gt_m6a[gt_m6a["gene_id_entrez"].isin(test_genes)]["log2fc"]
 
        fold_name = f"fold_{fold_idx + 1}"
        fold_data = {
            "test_chroms": sorted(test_chroms, key=_chrom_sort_key),
            "val_chroms": sorted(val_chroms, key=_chrom_sort_key),
            "train_chroms": sorted(train_chroms, key=_chrom_sort_key),
            "test_gene_ids": test_genes,
            "val_gene_ids": val_genes,
            "train_gene_ids": train_genes,
        }
 
        logger.info("-" * 60)
        logger.info("Fold %d:", fold_idx + 1)
        logger.info(
            "  test:  %4d genes (%5.1f%%)  %6d sites  [%s]",
            len(test_genes), 100 * len(test_genes) / total,
            n_test_sites, _fmt_chroms(test_chroms),
        )
        logger.info(
            "  val:   %4d genes (%5.1f%%)  %6d sites  [%s]",
            len(val_genes), 100 * len(val_genes) / total,
            n_val_sites, _fmt_chroms(val_chroms),
        )
        logger.info(
            "  train: %4d genes (%5.1f%%)  %6d sites  [%s]",
            len(train_genes), 100 * len(train_genes) / total,
            n_train_sites, _fmt_chroms(train_chroms),
        )
        logger.info(
            "  log2FC — train: %+.3f±%.3f  val: %+.3f±%.3f  test: %+.3f±%.3f",
            train_lfc.mean(), train_lfc.std(),
            val_lfc.mean(), val_lfc.std(),
            test_lfc.mean(), test_lfc.std(),
        )
        # 保存
        fold_path = output_dir / f"{fold_name}.json"
        with open(fold_path, "w") as f:
            json.dump(fold_data, f, indent=2)
        logger.info("  → %s", fold_path)
 
        all_folds[fold_name] = fold_data
        summary_stats["folds"][fold_name] = {
            "test_chroms": sorted(test_chroms, key=_chrom_sort_key),
            "val_chroms": sorted(val_chroms, key=_chrom_sort_key),
            "train_chroms": sorted(train_chroms, key=_chrom_sort_key),
            "n_test": len(test_genes),
            "n_val": len(val_genes),
            "n_train": len(train_genes),
            "pct_test": round(100 * len(test_genes) / total, 1),
            "pct_val": round(100 * len(val_genes) / total, 1),
            "pct_train": round(100 * len(train_genes) / total, 1),
            "n_test_sites": n_test_sites,
            "n_val_sites": n_val_sites,
            "n_train_sites": n_train_sites,
        }
 
    # =========================================================================
    # 6. 阴性对照集
    # =========================================================================
    neg_data = {
        "gene_ids": gt_neg["gene_id_entrez"].tolist(),
        "n_genes": len(gt_neg),
        "description": "无 m6A 位点的 protein-coding 基因 (空 bag 阴性对照)",
    }
    neg_path = output_dir / "negative_control.json"
    with open(neg_path, "w") as f:
        json.dump(neg_data, f, indent=2)
    logger.info("-" * 60)
    logger.info("阴性对照集: %d genes → %s", len(gt_neg), neg_path)
 
    # =========================================================================
    # 7. 汇总
    # =========================================================================
    summary_path = output_dir / "split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=2)
    logger.info("汇总 → %s", summary_path)
 
    # =========================================================================
    # 8. 比例汇总 + 覆盖率检查
    # =========================================================================
    logger.info("=" * 60)
    logger.info("各折比例汇总:")
    logger.info("  %-8s %7s %7s %7s", "Fold", "Train%", "Val%", "Test%")
    for fold_name in sorted(summary_stats["folds"]):
        f = summary_stats["folds"][fold_name]
        logger.info("  %-8s %6.1f%% %6.1f%% %6.1f%%",
                     fold_name, f["pct_train"], f["pct_val"], f["pct_test"])
 
    # 覆盖率
    all_test = set()
    for fold_data in all_folds.values():
        all_test.update(fold_data["test_gene_ids"])
    all_m6a = set(gt_m6a["gene_id_entrez"].tolist())
 
    covered = len(all_test & all_m6a)
    logger.info("-" * 60)
    logger.info("  test 总覆盖: %d / %d (%.1f%%)",
                covered, len(all_m6a), 100 * covered / len(all_m6a))
 
    # 检查 val 和 test 之间有没有重叠
    for i in range(n_folds):
        fi = all_folds[f"fold_{i+1}"]
        overlap = set(fi["test_gene_ids"]) & set(fi["val_gene_ids"])
        if overlap:
            logger.warning("  ⚠ Fold %d: test 和 val 有 %d 个基因重叠!", i+1, len(overlap))
 
    logger.info("=" * 60)
    logger.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
 
    ROOT = Path("/media/sda5/xsh-workspace/m6A_MIL")
 
    make_splits(
        gene_table_path=ROOT / "data/processed/gene_table.parquet",
        site_table_path=ROOT / "data/processed/site_table.parquet",
        output_dir=ROOT / "data/processed/splits",
        n_folds=5,
        target_val_pct=0.05,
        min_basemean=10,
    )
 