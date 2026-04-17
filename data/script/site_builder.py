from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class SiteTableBuilder:

    # -------------------------------------------------------------------------
    # Reader 家族分组 (两个功能阵营)
    # -------------------------------------------------------------------------
    #   YTHDF 阵营: 促进 mRNA 降解    → METTL3 KO 后该 mRNA 上调
    #   IGF2BP 阵营: 稳定 mRNA        → METTL3 KO 后该 mRNA 下调
    YTHDF_READERS = ("YTHDF1", "YTHDF2", "YTHDF3")
    IGF2BP_READERS = ("IGF2BP1", "IGF2BP2", "IGF2BP3")
    ALL_READERS = YTHDF_READERS + IGF2BP_READERS

    # -------------------------------------------------------------------------
    # HEK293 细胞系家族 (POSTAR3 的 sample 字段)
    # -------------------------------------------------------------------------
    # HEK293 母本 + 各种工具化衍生系 (T-antigen, FRT, SRRM4 转染等).
    # 这些差异主要在工具改造层面, 不影响 reader 的 RNA 结合特异性
    HEK293_SAMPLES = (
        "HEK293T",
        "HEK293",
        "HEK_293_FRT",
        "HEK293T,no_SRRM4",
        "HEK293T,plus_SRRM4",
    )
    # -------------------------------------------------------------------------
    # 标准染色体 (丢弃 alt / patch / scaffold)
    # -------------------------------------------------------------------------
    STANDARD_CHROMS = frozenset(
        {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY", "chrM"}
    )
    DENSITY_WINDOW = 500
    MOESM4_SIMETTL3_SHEET = "siMETTL3 common sites"

    def __init__(
        self,
        gene_table_path,
        m6a_wt_xlsx,       # MOESM3: WT HEK293T 所有 m6A 位点
        m6a_sirna_xlsx,    # MOESM4: siMETTL3 等处理后的残留位点 (多 sheet)
        rbp_txt_gz,        # POSTAR3: 人类 RBP peaks (多细胞系, 需过滤)
        gtf_path,          # Ensembl GTF (做 UTR/CDS 分类)
        output_path=None,
    ):
        self.gene_table_path = Path(gene_table_path)
        self.m6a_wt_xlsx = Path(m6a_wt_xlsx)
        self.m6a_sirna_xlsx = Path(m6a_sirna_xlsx)
        self.rbp_txt_gz = Path(rbp_txt_gz)
        self.gtf_path = Path(gtf_path)
        self.output_path = Path(output_path) if output_path else None

        # 中间结果 
        self.gene_table_: Optional[pd.DataFrame] = None
        self.m6a_sites_: Optional[pd.DataFrame] = None             # Step 2: 纯 m6A 信息
        self.sites_with_genes_: Optional[pd.DataFrame] = None      # Step 3: + gene 映射
        self.rbp_peaks_: Optional[dict] = None                      # Step 4: {reader: peaks_df}
        self.sites_with_binding_: Optional[pd.DataFrame] = None    # Step 5: + 6 列 reader bool
        self.site_table_: Optional[pd.DataFrame] = None             # Step 8: 最终产物

    def __repr__(self) -> str:
        mark = lambda x: "✓" if x is not None else "·"
        return (
            f"SiteTableBuilder(\n"
            f"  gene_table         [{mark(self.gene_table_)}]\n"
            f"  m6a_sites          [{mark(self.m6a_sites_)}]\n"
            f"  sites_with_genes   [{mark(self.sites_with_genes_)}]\n"
            f"  rbp_peaks          [{mark(self.rbp_peaks_)}]\n"
            f"  sites_with_binding [{mark(self.sites_with_binding_)}]\n"
            f"  site_table         [{mark(self.site_table_)}]\n"
            f")"
        )

    # =========================================================================
    # Step 1: 读 gene_table (作为 gene 级元数据源)
    # =========================================================================
    def load_gene_table(self) -> pd.DataFrame:
        logger.info("[1/8] Loading gene_table from %s", self.gene_table_path)
        df = pd.read_parquet(self.gene_table_path)
        # gene_table parquet 的 index 是 gene_id_entrez; reset 成列便于后续 overlap
        if df.index.name == "gene_id_entrez":
            df = df.reset_index()

        required = [
            "gene_id_entrez", "gene_id_ensembl", "symbol",
            "chrom", "start", "end", "strand",
            "mane_transcript_id",
        ]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            raise ValueError(f"gene_table 缺少必需列: {missing_cols}")

        logger.info("  gene_table 总行数: %d", len(df))
        logger.info(
            "  其中有 MANE transcript 的: %d (%.1f%%)",
            int(df["mane_transcript_id"].notna().sum()),
            100 * df["mane_transcript_id"].notna().mean(),
        )

        self.gene_table_ = df
        return df

    # =========================================================================
    # Step 2: 读 m6A 位点 (MOESM3 WT + MOESM4 siMETTL3)
    # =========================================================================
    def load_m6a_sites(self) -> pd.DataFrame:
        logger.info("[2/8] Loading m6A sites from GLORI...")
        # --- 读 MOESM3 (WT) ---
        wt = pd.read_excel(self.m6a_wt_xlsx)
        logger.info("  MOESM3 (WT) 原始行数: %d", len(wt))
        wt = self._normalize_m6a_cols(wt)
        wt["m6a_level_wt"] = self._weighted_level(wt)

        # --- 读 MOESM4 siMETTL3 子表 ---
        sirna = pd.read_excel(self.m6a_sirna_xlsx, 
                            sheet_name=self.MOESM4_SIMETTL3_SHEET,
                            skiprows=[1])
        logger.info(
            "  MOESM4 '%s' 原始行数: %d",
            self.MOESM4_SIMETTL3_SHEET, len(sirna),
        )
        sirna = self._normalize_m6a_cols(sirna)
        sirna["m6a_level_sirna"] = self._weighted_level(sirna)

         # --- Merge: 以 WT 为基准, 左 join siMETTL3 ---
        # 主键: (chrom, pos, strand) 三元组精确定位
        key = ["chrom", "pos", "strand"]
        merged = wt.merge(
            sirna[key + ["m6a_level_sirna"]],
            on=key,
            how="left",
        )
        # 不在 siMETTL3 表里的位点 → NaN → 填 0.0
        # 语义: "siMETTL3 后完全消失" = METTL3 100% 负责这个位点
        merged["m6a_level_sirna"] = merged["m6a_level_sirna"].fillna(0.0)

        # METTL3 依赖部分 (clamp 到 0: 防止测量噪声导致 sirna > wt)
        merged["m6a_level_mettl3_dep"] = (
            (merged["m6a_level_wt"] - merged["m6a_level_sirna"]).clip(lower=0.0)
        )
        cols_keep = [
            "chrom", "pos", "strand",
            "m6a_level_wt", "m6a_level_sirna", "m6a_level_mettl3_dep",
            "agcov_rep1", "agcov_rep2",
        ]
        if "gene_symbol_glori" in merged.columns:
            cols_keep.append("gene_symbol_glori")
        if "cluster_info" in merged.columns:
            cols_keep.append("cluster_info")

        merged = merged[[c for c in cols_keep if c in merged.columns]]
        # --- 体检 ---
        n_total = len(merged)
        n_in_sirna = int((merged["m6a_level_sirna"] > 0).sum())
        n_fully_dep = int((merged["m6a_level_sirna"] == 0).sum())

        logger.info("-" * 60)
        logger.info("  WT m6A 位点总数: %d", n_total)
        logger.info(
            "  └─ siMETTL3 仍存在 (METTL3 非全责): %d (%.1f%%)",
            n_in_sirna, 100 * n_in_sirna / n_total,
        )
        logger.info(
            "  └─ siMETTL3 完全消失 (METTL3 全责): %d (%.1f%%)",
            n_fully_dep, 100 * n_fully_dep / n_total,
        )
        logger.info("  m6a_level_wt:          median=%.3f  mean=%.3f",
                    merged["m6a_level_wt"].median(),
                    merged["m6a_level_wt"].mean())
        logger.info("  m6a_level_sirna:       median=%.3f  mean=%.3f",
                    merged["m6a_level_sirna"].median(),
                    merged["m6a_level_sirna"].mean())
        logger.info("  m6a_level_mettl3_dep:  median=%.3f  mean=%.3f",
                    merged["m6a_level_mettl3_dep"].median(),
                    merged["m6a_level_mettl3_dep"].mean())

        self.m6a_sites_ = merged
        return merged

    # =========================================================================
    # Step 3: m6A 位点 -> 基因映射 (pyranges strand-matched overlap)
    # =========================================================================
    def assign_sites_to_genes(self) -> pd.DataFrame:
        if self.m6a_sites_ is None:
            self.load_m6a_sites()
        if self.gene_table_ is None:
            self.load_gene_table()
        logger.info("[3/8] Assigning m6A sites to genes (pyranges overlap)...")

        import pyranges as pr
        # GLORI 的 pos 是 1-based 单碱基. 转 BED 0-based half-open:
        #   Start = pos - 1,  End = pos
        sites = self.m6a_sites_.reset_index(drop=True).copy()
        sites_pr = pr.PyRanges(pd.DataFrame({
            "Chromosome": sites["chrom"],
            "Start": sites["pos"] - 1,
            "End": sites["pos"],
            "Strand": sites["strand"],
            "_site_idx": sites.index,
        }))
        # --- 构造 genes PyRanges ---
        # 只保留有 MANE 的 gene (后续 UTR/CDS 分类需要).
        # NCBI annot 的 start/end 是 1-based inclusive, 转 BED: Start = start - 1, End = end.
        genes = self.gene_table_.copy()
        genes = genes[genes["mane_transcript_id"].notna()].reset_index(drop=True)
        logger.info("  用于 overlap 的 gene 数 (有 MANE): %d", len(genes))
        genes_pr = pr.PyRanges(pd.DataFrame({
            "Chromosome": genes["chrom"],
            "Start": genes["start"] - 1,
            "End": genes["end"],
            "Strand": genes["strand"],
            "_gene_idx": genes.index,
        }))

        joined = sites_pr.join(genes_pr, strandedness="same").df
        if len(joined) == 0:
            raise RuntimeError(
                "m6A × gene overlap 结果为空! 检查 chrom 命名是否一致 "
                "(都应是 chr1 风格)"
            )
        mapping = joined[["_site_idx", "_gene_idx"]].copy()
        out = mapping.merge(
            sites, left_on="_site_idx", right_index=True, how="left",
        )
        gene_meta_cols = [
            "gene_id_entrez", "gene_id_ensembl", "symbol", "mane_transcript_id",
        ]
        if "is_par" in genes.columns:
            gene_meta_cols.append("is_par")
        gene_meta = genes[gene_meta_cols]

        out = out.merge(
            gene_meta, left_on="_gene_idx", right_index=True, how="left",
        )
        out = out.drop(columns=["_site_idx", "_gene_idx"]).reset_index(drop=True)


        n_sites_in = len(self.m6a_sites_)
        n_sites_unique = out[["chrom", "pos", "strand"]].drop_duplicates().shape[0]
        n_rows = len(out)
        multi_gene_sites = int(
            (out.groupby(["chrom", "pos", "strand"]).size() > 1).sum()
        )
        sites_per_gene = out.groupby("gene_id_entrez").size()

        logger.info("-" * 60)
        logger.info("  输入 m6A 位点 (唯一): %d", n_sites_in)
        logger.info(
            "  映射到 protein-coding gene 的位点 (唯一): %d (%.1f%%)",
            n_sites_unique, 100 * n_sites_unique / n_sites_in,
        )
        logger.info(
            "  丢弃 (intergenic / 仅 overlap 非 MANE gene): %d (%.1f%%)",
            n_sites_in - n_sites_unique,
            100 * (n_sites_in - n_sites_unique) / n_sites_in,
        )
        logger.info(
            "  输出 (site, gene) 组合行数: %d  [比位点数多 %d, 重叠基因所致]",
            n_rows, n_rows - n_sites_unique,
        )
        logger.info(
            "  同时映射到多个 gene 的位点: %d (%.2f%%)",
            multi_gene_sites, 100 * multi_gene_sites / max(n_sites_unique, 1),
        )
        logger.info(
            "  每个 gene 上的 m6A 位点数: median=%d  mean=%.1f  max=%d",
            int(sites_per_gene.median()),
            sites_per_gene.mean(),
            int(sites_per_gene.max()),
        )

        # sanity check: GLORI 自带 Gene symbol 和 overlap 结果一致率
        # 不一致主要在GLORI里记录的ELSE
        if "gene_symbol_glori" in out.columns:
            has_glori = out["gene_symbol_glori"].notna()
            if has_glori.any():
                matched = (out.loc[has_glori, "gene_symbol_glori"]
                           == out.loc[has_glori, "symbol"]).sum()
                total = int(has_glori.sum())
                logger.info(
                    "  sanity check: GLORI Gene vs overlap 结果一致: %d/%d (%.1f%%)",
                    matched, total, 100 * matched / total,
                )

        # 落到 PAR 基因上的位点统计 (如果 gene_table 有 is_par)
        if "is_par" in out.columns:
            n_par_rows = int(out["is_par"].fillna(False).sum())
            if n_par_rows > 0:
                logger.info(
                    "  落在 PAR 基因上的 (site, gene) 行数: %d", n_par_rows,
                )

        self.sites_with_genes_ = out
        return out
    
    # =========================================================================
    # Step 4: 读 POSTAR3 RBP peaks
    # =========================================================================
    def load_rbp_peaks(self) -> dict:
        logger.info("[4/8] Loading RBP peaks from %s", self.rbp_txt_gz)
        cols = ["chrom", "start", "end", "peak_id", "strand",
                "rbp", "method", "sample", "accession", "score"]
        dtype = {
            "chrom": str, "start": np.int64, "end": np.int64,
            "peak_id": str, "strand": str, "rbp": str,
            "method": str, "sample": str, "accession": str,
            "score": str,   # 跨方法格式不一, str 最安全
        }
        peaks = pd.read_csv(
            self.rbp_txt_gz,
            sep="\t", names=cols, dtype=dtype,
            compression="gzip", low_memory=False,
        )
        logger.info("  POSTAR3 总 peak 数: %d", len(peaks))
        peaks = peaks[peaks["rbp"].isin(self.ALL_READERS)].copy()
        logger.info("  筛 YTH/IGF2BP 家族后: %d", len(peaks))
        # --- 过滤 2: HEK293 细胞系家族 ---
        before_cell = len(peaks)
        peaks = peaks[peaks["sample"].isin(self.HEK293_SAMPLES)].copy()
        logger.info(
            "  筛 HEK293 细胞系家族后: %d (丢弃 %d, %.1f%%)",
            len(peaks), before_cell - len(peaks),
            100 * (before_cell - len(peaks)) / max(before_cell, 1),
        )
        # --- 过滤 3: 标准染色体 ---
        peaks = peaks[peaks["chrom"].isin(self.STANDARD_CHROMS)].copy()
        logger.info("  筛标准染色体后: %d", len(peaks))

        # --- 按 reader 拆分, 打印每个的 peak / method / sample 分布 ---
        logger.info("-" * 60)
        logger.info("  各 reader peak 分布 (过滤后):")
        result = {}
        for reader in self.ALL_READERS:
            sub = peaks[peaks["rbp"] == reader].copy()
            n_peaks = len(sub)
            n_methods = sub["method"].nunique()
            n_samples = sub["sample"].nunique()
            logger.info(
                "    %-10s n=%6d  methods=%d  samples=%d",
                reader, n_peaks, n_methods, n_samples,
            )
            if n_peaks > 0:
                method_counts = sub["method"].value_counts().head(3)
                logger.info(
                    "        主要方法: %s",
                    ", ".join(f"{m}({c})" for m, c in method_counts.items()),
                )
            result[reader] = sub[
                ["chrom", "start", "end", "strand", "method", "sample"]
            ].reset_index(drop=True)

        self.rbp_peaks_ = result
        return result
    
    # =========================================================================
    # Step 5: 标注 m6A 位点 × reader 结合 (6 列 bool)
    # =========================================================================
    def annotate_reader_binding(self) -> pd.DataFrame:
        if self.sites_with_genes_ is None:
            self.assign_sites_to_genes()
        if self.rbp_peaks_ is None:
            self.load_rbp_peaks()
        logger.info("[5/8] Annotating reader binding (6 readers)...")

        import pyranges as pr

        sites = self.sites_with_genes_
        unique_sites = (
            sites[["chrom", "pos", "strand"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        unique_sites["_usite_idx"] = unique_sites.index
        sites_pr = pr.PyRanges(pd.DataFrame({
            "Chromosome": unique_sites["chrom"],
            "Start": unique_sites["pos"] - 1,
            "End": unique_sites["pos"],
            "Strand": unique_sites["strand"],
            "_usite_idx": unique_sites["_usite_idx"],
        }))

        logger.info("  唯一 m6A 位点数 (参与 overlap): %d", len(unique_sites))
        logger.info("-" * 60)
        # --- 对每个 reader 做 overlap ---
        binding_cols = {}
        for reader in self.ALL_READERS:
            peaks_df = self.rbp_peaks_[reader]
            col_name = f"{reader.lower()}_bound"

            if len(peaks_df) == 0:
                binding_cols[col_name] = np.zeros(len(unique_sites), dtype=bool)
                logger.info(
                    "    %-10s : 0 sites bound (no peaks available)", reader,
                )
                continue
            buffer= 20
            peaks_pr = pr.PyRanges(pd.DataFrame({
                "Chromosome": peaks_df["chrom"],
                "Start": (peaks_df["start"] - buffer).clip(lower=0),
                "End": peaks_df["end"] + buffer,
                "Strand": peaks_df["strand"],
            }))

            overlapped = sites_pr.overlap(peaks_pr, strandedness="same").df

            if len(overlapped) > 0:
                bound_idx = set(overlapped["_usite_idx"].values)
            else:
                bound_idx = set()
            bound_mask = unique_sites["_usite_idx"].isin(bound_idx).to_numpy()
            binding_cols[col_name] = bound_mask

            n_bound = int(bound_mask.sum())
            logger.info(
                "    %-10s : %6d / %d sites bound (%.1f%%)",
                reader, n_bound, len(unique_sites),
                100 * n_bound / len(unique_sites),
            )

        # --- 组装唯一位点 + 6 列 bool ---
        binding_df = pd.DataFrame(binding_cols)
        unique_sites_binding = pd.concat(
            [unique_sites.drop(columns=["_usite_idx"]), binding_df],
            axis=1,
        )

        # --- 广播回 (site, gene) 行 ---
        out = sites.merge(
            unique_sites_binding,
            on=["chrom", "pos", "strand"],
            how="left",
        )
        # --- 家族级 / 跨家族统计 (基于唯一位点, 避免重复基因放大) ---
        logger.info("-" * 60)
        bound_cols = [f"{r.lower()}_bound" for r in self.ALL_READERS]
        ythdf_cols = [f"{r.lower()}_bound" for r in self.YTHDF_READERS]
        igf2bp_cols = [f"{r.lower()}_bound" for r in self.IGF2BP_READERS]

        any_bound = unique_sites_binding[bound_cols].any(axis=1)
        ythdf_any = unique_sites_binding[ythdf_cols].any(axis=1)
        igf2bp_any = unique_sites_binding[igf2bp_cols].any(axis=1)
        both_any = ythdf_any & igf2bp_any

        n_uniq = len(unique_sites_binding)
        logger.info(
            "  至少一个 reader 结合:      %6d (%.1f%%)",
            int(any_bound.sum()), 100 * any_bound.mean(),
        )
        logger.info(
            "  YTHDF 家族任一结合:        %6d (%.1f%%)",
            int(ythdf_any.sum()), 100 * ythdf_any.mean(),
        )
        logger.info(
            "  IGF2BP 家族任一结合:       %6d (%.1f%%)",
            int(igf2bp_any.sum()), 100 * igf2bp_any.mean(),
        )
        logger.info(
            "  两家族都结合 (竞争位点):   %6d (%.1f%%)",
            int(both_any.sum()), 100 * both_any.mean(),
        )

        self.sites_with_binding_ = out
        return out

    # =========================================================================
    # Step 6: 转录本区段分类 (5'UTR / CDS / 3'UTR / intron)
    # =========================================================================
    def classify_region(self) -> pd.DataFrame:
        if self.sites_with_binding_ is None:
            self.annotate_reader_binding()

        logger.info("[6/8] Classifying transcript region (5UTR/CDS/3UTR/intron)...")
        # --- Step 6a: 解析 GTF, 提取 MANE transcript 的 exon 和 CDS ---
        # 收集需要的 MANE transcript IDs (去版本号后的)
        needed_tx = set(
            self.sites_with_binding_["mane_transcript_id"].dropna().unique()
        )
        logger.info("  需要解析的 MANE transcript 数: %d", len(needed_tx))
        # 从 GTF 提取 exon 和 CDS 行
        # 存储: tx_id → {"exons": [(start, end), ...], "cds": [(start, end), ...], "strand": "+"/"-"}
        tx_structure = {}
        opener = gzip.open(self.gtf_path, "rt") if str(self.gtf_path).endswith(".gz") else open(self.gtf_path, "rt")
        import re
        attr_gene_re = re.compile(r'transcript_id "([^"]+)"')

        with opener as f:
            for line in f:
                if line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 9:
                    continue
                feature = fields[2]
                if feature not in ("exon", "CDS"):
                    continue

                attrs = fields[8]
                m = attr_gene_re.search(attrs)
                if m is None:
                    continue
                tx_id_raw = m.group(1)
                tx_id = tx_id_raw.split(".")[0]  # 去版本号

                if tx_id not in needed_tx:
                    continue

                # GTF 是 1-based inclusive; 转 0-based half-open (和 pyranges 一致)
                start_0 = int(fields[3]) - 1
                end_0 = int(fields[4])
                strand = fields[6]

                if tx_id not in tx_structure:
                    tx_structure[tx_id] = {"exons": [], "cds": [], "strand": strand}

                if feature == "exon":
                    tx_structure[tx_id]["exons"].append((start_0, end_0))
                elif feature == "CDS":
                    tx_structure[tx_id]["cds"].append((start_0, end_0))

        logger.info("  GTF 中找到结构信息的 transcript: %d / %d",
                    len(tx_structure), len(needed_tx))
        # 预计算每个 transcript 的 CDS 整体范围 (min_start, max_end)
        # 用于快速判断 5UTR/CDS/3UTR
        tx_cds_range = {}
        for tx_id, info in tx_structure.items():
            if info["cds"]:
                all_cds_starts = [s for s, e in info["cds"]]
                all_cds_ends = [e for s, e in info["cds"]]
                tx_cds_range[tx_id] = (min(all_cds_starts), max(all_cds_ends))
            # 没有 CDS 的 transcript (非编码? 不应该出现在 MANE protein-coding 里, 但防御)

        # 把 exon 列表转成 sorted tuple 方便查询
        tx_exon_sets = {}
        for tx_id, info in tx_structure.items():
            # 排序后的 exon list
            tx_exon_sets[tx_id] = sorted(info["exons"], key=lambda x: x[0])

        # --- Step 6b: 对每个 (site, gene) 行分类 ---
        sites = self.sites_with_binding_.copy()

        def _classify_one(row):
            """对单行 (site, gene) 做区段分类."""
            tx_id = row.get("mane_transcript_id")
            if pd.isna(tx_id) or tx_id not in tx_structure:
                return "unknown"

            pos_0 = int(row["pos"]) - 1  # 转 0-based (和 GTF exon/CDS 对齐)
            strand = tx_structure[tx_id]["strand"]
            exons = tx_exon_sets[tx_id]

            # 1. 检查是否在 exon 内
            in_exon = False
            for ex_s, ex_e in exons:
                if ex_s <= pos_0 < ex_e:
                    in_exon = True
                    break
            if not in_exon:
                return "intron"

            # 2. 在 exon 内, 检查相对 CDS 的位置
            if tx_id not in tx_cds_range:
                return "unknown"  # 没有 CDS 注释

            cds_start, cds_end = tx_cds_range[tx_id]

            if cds_start <= pos_0 < cds_end:
                # 进一步确认: pos 真的在某个 CDS exon 内 (不是 CDS 范围内的 intron)
                # 但这个情况很罕见 (已经检查过 in_exon), 而且 CDS 内 intron 不存在
                return "CDS"

            # 根据 strand 判断 5UTR / 3UTR
            if strand == "+":
                if pos_0 < cds_start:
                    return "5UTR"
                else:  # pos_0 >= cds_end
                    return "3UTR"
            else:  # strand == "-"
                # 负链: 基因组高坐标端是 5'端
                if pos_0 >= cds_end:
                    return "5UTR"
                else:  # pos_0 < cds_start
                    return "3UTR"

        logger.info("  正在分类 %d 行 (可能需要 10-30 秒)...", len(sites))
        sites["region"] = sites.apply(_classify_one, axis=1)

        # --- 体检 ---
        region_counts = sites["region"].value_counts()
        logger.info("-" * 60)
        logger.info("  区段分类结果:")
        for region in ["5UTR", "CDS", "3UTR", "intron", "unknown"]:
            n = region_counts.get(region, 0)
            logger.info("    %-8s: %6d (%.1f%%)", region, n, 100 * n / len(sites))

        # 预期: 3UTR > CDS > 5UTR > intron (m6A 经典分布)
        if region_counts.get("3UTR", 0) > 0 and region_counts.get("CDS", 0) > 0:
            ratio_3utr = region_counts.get("3UTR", 0) / len(sites)
            logger.info(
                "  ^ m6A 经典分布: 3'UTR 应最多 (>40%%), 你的: %.1f%%",
                100 * ratio_3utr,
            )

        self.sites_with_binding_ = sites
        return sites


    # =========================================================================
    # Step 7: m6A 局部密度
    # m6A 位点在 mRNA 上有明显聚集现象 (clustering).
    # 聚集位点倾向于功能性更强 (多个 m6A 协同招募 reader, 形成液-液相分离).
    # 孤立位点倾向于功能沉默.
    # =========================================================================
    def compute_m6a_density(self) -> pd.DataFrame:
        if self.sites_with_binding_ is None:
            self.annotate_reader_binding()

        logger.info("[7/8] Computing m6A density (±%dnt)...", self.DENSITY_WINDOW)

        from bisect import bisect_left, bisect_right

        sites = self.sites_with_binding_.copy()
        m6a_all = self.m6a_sites_[["chrom", "pos", "strand"]].copy()
        # 按 (chrom, strand) 分组, 每组内 sort by pos
        pos_index = {}  # key: (chrom, strand), value: sorted list of positions
        for (chrom, strand), grp in m6a_all.groupby(["chrom", "strand"]):
            pos_index[(chrom, strand)] = sorted(grp["pos"].values)
        window = self.DENSITY_WINDOW
        def _density(row):
            key = (row["chrom"], row["strand"])
            if key not in pos_index:
                return 0
            positions = pos_index[key]
            p = int(row["pos"])
            # bisect 找 [p - window, p + window] 范围内的位点数
            left = bisect_left(positions, p - window)
            right = bisect_right(positions, p + window)
            count = right - left - 1  # 减去自己
            return max(count, 0)  # 防御: 理论上不应为负

        logger.info("  正在计算 %d 行的 density (bisect)...", len(sites))
        sites["m6a_density_500nt"] = sites.apply(_density, axis=1)
        # --- 体检 ---
        density = sites["m6a_density_500nt"]
        logger.info("-" * 60)
        logger.info("  m6A density (±%dnt) 分布:", self.DENSITY_WINDOW)
        logger.info("    mean=%.1f  median=%d  max=%d",
                    density.mean(), int(density.median()), int(density.max()))
        logger.info("    = 0 (孤立位点):   %6d (%.1f%%)",
                    int((density == 0).sum()), 100 * (density == 0).mean())
        logger.info("    1-3 (稀疏):       %6d (%.1f%%)",
                    int(((density >= 1) & (density <= 3)).sum()),
                    100 * ((density >= 1) & (density <= 3)).mean())
        logger.info("    4-10 (中等聚集):  %6d (%.1f%%)",
                    int(((density >= 4) & (density <= 10)).sum()),
                    100 * ((density >= 4) & (density <= 10)).mean())
        logger.info("    > 10 (密集聚集):  %6d (%.1f%%)",
                    int((density > 10).sum()), 100 * (density > 10).mean())

        self.sites_with_binding_ = sites
        return sites

    # =========================================================================
    # Step 8: 组装最终表 + 保存
    # =========================================================================
    def build_table(self) -> pd.DataFrame:
        """串联所有步骤, 组装最终 site_table, 保存 parquet."""

        # 确保前置步骤都跑过
        if self.gene_table_ is None:
            self.load_gene_table()
        if self.m6a_sites_ is None:
            self.load_m6a_sites()
        if self.sites_with_genes_ is None:
            self.assign_sites_to_genes()
        if self.rbp_peaks_ is None:
            self.load_rbp_peaks()
        if self.sites_with_binding_ is None:
            self.annotate_reader_binding()

        # Step 6 和 7 可能还没跑 (如果用户直接调 build_table)
        if "region" not in self.sites_with_binding_.columns:
            self.classify_region()
        if "m6a_density_500nt" not in self.sites_with_binding_.columns:
            self.compute_m6a_density()

        logger.info("[8/8] Assembling final site_table...")

        table = self.sites_with_binding_.copy()

        # --- 列顺序整理 ---
        reader_cols = [f"{r.lower()}_bound" for r in self.ALL_READERS]

        col_order = [
            # 位点身份
            "chrom", "pos", "strand",
            # 基因映射
            "gene_id_entrez", "gene_id_ensembl", "symbol", "mane_transcript_id",
            # m6A 定量
            "m6a_level_wt", "m6a_level_sirna", "m6a_level_mettl3_dep",
            "agcov_rep1", "agcov_rep2",
            # reader 结合
            *reader_cols,
            # 位置 + 上下文
            "region", "m6a_density_500nt",
        ]
        # 可选列 (有就带, 没有就跳)
        optional = ["gene_symbol_glori", "cluster_info", "is_par"]
        for c in optional:
            if c in table.columns:
                col_order.append(c)

        # 只保留存在的列 (防御性)
        col_order = [c for c in col_order if c in table.columns]
        table = table[col_order]

        # --- 最终体检 ---
        self._print_site_stats(table)

        self.site_table_ = table
        return table

    def _print_site_stats(self, table: pd.DataFrame) -> None:
        """site_table 最终体检."""
        n = len(table)
        n_genes = table["gene_id_entrez"].nunique()
        n_unique_sites = table[["chrom", "pos", "strand"]].drop_duplicates().shape[0]

        reader_cols = [f"{r.lower()}_bound" for r in self.ALL_READERS]
        any_reader = table[reader_cols].any(axis=1)

        logger.info("=" * 60)
        logger.info("site_table 构建完成")
        logger.info("  总行数 (site, gene):       %d", n)
        logger.info("  唯一 m6A 位点:             %d", n_unique_sites)
        logger.info("  涉及 gene 数:              %d", n_genes)
        logger.info("  有 reader 结合的行:        %d (%.1f%%)",
                    int(any_reader.sum()), 100 * any_reader.mean())
        if "region" in table.columns:
            logger.info("  区段分布: %s",
                        table["region"].value_counts().to_dict())
        if "m6a_density_500nt" in table.columns:
            logger.info("  density 均值: %.1f  中位数: %d",
                        table["m6a_density_500nt"].mean(),
                        int(table["m6a_density_500nt"].median()))
        logger.info("=" * 60)

    def save(self) -> None:
        if self.output_path is None:
            raise ValueError("output_path 未设置")
        if self.site_table_ is None:
            raise ValueError("site_table_ 尚未构建, 先调用 build_table()")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.site_table_.to_parquet(self.output_path, index=False)
        logger.info("Saved site_table to %s  (%d rows)", self.output_path, len(self.site_table_))
    
    # =========================================================================
    # Step 9 (可选): 回填 gene_table 的统计列
    # =========================================================================
    def backfill_gene_table(self, gene_table_path=None) -> pd.DataFrame:
        """
        用 site_table 聚合统计, 更新 gene_table 的:
          n_m6a_sites:   该 gene 上的 m6A 位点数
          n_ythdf_sites: 该 gene 上被 YTHDF 家族任一结合的 m6A 位点数
          n_igf2bp_sites:该 gene 上被 IGF2BP 家族任一结合的 m6A 位点数
        """
        if self.site_table_ is None:
            raise ValueError("先调用 build_table()")

        logger.info("[9] Backfilling gene_table with site statistics...")

        path = Path(gene_table_path) if gene_table_path else self.gene_table_path
        gt = pd.read_parquet(path)

        st = self.site_table_

        # --- 每个 gene 的 m6A 位点数 ---
        n_m6a = st.groupby("gene_id_entrez").size().rename("n_m6a_sites")

        # --- 每个 gene 的 YTHDF 结合位点数 ---
        ythdf_cols = [f"{r.lower()}_bound" for r in self.YTHDF_READERS]
        st_ythdf = st[st[ythdf_cols].any(axis=1)]
        n_ythdf = st_ythdf.groupby("gene_id_entrez").size().rename("n_ythdf_sites")

        # --- 每个 gene 的 IGF2BP 结合位点数 ---
        igf2bp_cols = [f"{r.lower()}_bound" for r in self.IGF2BP_READERS]
        st_igf2bp = st[st[igf2bp_cols].any(axis=1)]
        n_igf2bp = st_igf2bp.groupby("gene_id_entrez").size().rename("n_igf2bp_sites")

        # --- 合并回 gene_table ---
        # gene_table 的 index 可能是 gene_id_entrez
        if gt.index.name == "gene_id_entrez":
            gt["n_m6a_sites"] = n_m6a.reindex(gt.index).fillna(0).astype("Int64")
            gt["n_ythdf_sites"] = n_ythdf.reindex(gt.index).fillna(0).astype("Int64")
            gt["n_igf2bp_sites"] = n_igf2bp.reindex(gt.index).fillna(0).astype("Int64")
        else:
            gt = gt.set_index("gene_id_entrez")
            gt["n_m6a_sites"] = n_m6a.reindex(gt.index).fillna(0).astype("Int64")
            gt["n_ythdf_sites"] = n_ythdf.reindex(gt.index).fillna(0).astype("Int64")
            gt["n_igf2bp_sites"] = n_igf2bp.reindex(gt.index).fillna(0).astype("Int64")

        # --- 体检 ---
        has_m6a = gt["n_m6a_sites"] > 0
        logger.info("  有 m6A 位点的 gene:     %d (%.1f%%)",
                    int(has_m6a.sum()), 100 * has_m6a.mean())
        logger.info("  有 YTHDF 结合的 gene:   %d", int((gt["n_ythdf_sites"] > 0).sum()))
        logger.info("  有 IGF2BP 结合的 gene:  %d", int((gt["n_igf2bp_sites"] > 0).sum()))
        logger.info("  每 gene m6A 位点数: mean=%.1f  median=%d  max=%d",
                    gt.loc[has_m6a, "n_m6a_sites"].mean(),
                    int(gt.loc[has_m6a, "n_m6a_sites"].median()),
                    int(gt["n_m6a_sites"].max()))

        # --- 覆盖保存 ---
        gt.to_parquet(path)
        logger.info("  已更新 gene_table: %s", path)

        return gt
    
    @staticmethod
    def _normalize_m6a_cols(df: pd.DataFrame) -> pd.DataFrame:
        
        return df.rename(columns={
            "Chr": "chrom",
            "Sites": "pos",
            "Strand": "strand",
            "Gene": "gene_symbol_glori",    # 留作 sanity check, 不作为主映射
            "AGcov_rep1": "agcov_rep1",     # MOESM3 写法
            "AGCov_rep1": "agcov_rep1",     # MOESM4 写法
            "AGcov_rep2": "agcov_rep2",
            "AGCov_rep2": "agcov_rep2",
            "m6A_level_rep1": "level_rep1",
            "m6A_level_rep2": "level_rep2",
        })
    @staticmethod
    def _weighted_level(df: pd.DataFrame) -> pd.Series:
        
        numer = (df["level_rep1"] * df["agcov_rep1"]
                 + df["level_rep2"] * df["agcov_rep2"])
        denom = df["agcov_rep1"] + df["agcov_rep2"]
       
        with np.errstate(invalid="ignore", divide="ignore"):
            level = (numer / denom).clip(0.0, 1.0)
        return level
    
    def run_all(self) -> pd.DataFrame:
        """一键执行全部 pipeline (Step 1-9)."""
        self.load_gene_table()
        self.load_m6a_sites()
        self.assign_sites_to_genes()
        self.load_rbp_peaks()
        self.annotate_reader_binding()
        self.classify_region()
        self.compute_m6a_density()
        self.build_table()
        if self.output_path is not None:
            self.save()
        self.backfill_gene_table()
        return self.site_table_
    
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    ROOT = Path("/media/sda5/xsh-workspace/m6A_MIL")

    builder = SiteTableBuilder(
        gene_table_path=ROOT / "data/processed/gene_table.parquet",
        m6a_wt_xlsx=ROOT / "data/raw/m6a/41587_2022_1487_MOESM3_ESM.xlsx",
        m6a_sirna_xlsx=ROOT / "data/raw/m6a/41587_2022_1487_MOESM4_ESM.xlsx",
        rbp_txt_gz=ROOT / "data/raw/rbp/human.txt.gz",
        gtf_path=ROOT / "data/raw/Homo_sapiens.GRCh38.115.chr.gtf.gz",
        output_path=ROOT / "data/processed/site_table.parquet",
    )
    builder.run_all()
    print(builder)
    
    