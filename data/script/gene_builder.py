


from __future__ import annotations
 
import gzip
import logging
import re
from pathlib import Path
from typing import Optional
 
import pandas as pd
 
logger = logging.getLogger(__name__)

class GeneTableBuilder:
    # -------------------------------------------------------------------------
    # 样本分组 (GSE182607)
    # -------------------------------------------------------------------------
    # 16 个 GSM 的完整构成:
    #   sgNS / sgMETTL3 / sgMETTL14 / sgMETTL16 共 4 组,
    #   每组 2 个生物学重复 × (Input + m6A-IP) = 4 samples.
    # 本研究只要 sgNS vs sgMETTL3 的 Input (RNA-seq) 样本:

    WT_SAMPLES = ["GSM5532219", "GSM5532221"] 
    KO_SAMPLES = ["GSM5532223", "GSM5532225"] 

    # -------------------------------------------------------------------------
    # 染色体命名映射: NCBI RefSeq accession -> UCSC/Ensembl 风格
    # -------------------------------------------------------------------------
    # annot 表用 NC_000001.11 这种 RefSeq accession,
    # GLORI / POSTAR3 / 主流 GTF 都用 chr1. 统一到 chr1 风格.
    # 版本号 (`.11`, `.12`) 在不同染色体上不同, 按前缀 (NC_000001) 匹配即可.
    _REFSEQ_CHROM_MAP: dict = {
        **{f"NC_{i:06d}": f"chr{i}" for i in range(1, 23)},  # chr1..chr22
        "NC_000023": "chrX",
        "NC_000024": "chrY",
        "NC_012920": "chrM",
    }

    # -------------------------------------------------------------------------
    # GTF 属性正则 (预编译, 重复调用时加速)
    # -------------------------------------------------------------------------
    _ATTR_RE_CACHE: dict = {}

    def __init__(
        self,
        counts_path,
        annot_path,
        gtf_path,
        output_path=None,
    ):
        self.counts_path = Path(counts_path)
        self.annot_path = Path(annot_path)
        self.gtf_path = Path(gtf_path)
        self.output_path = Path(output_path) if output_path else None

        # 中间结果 (trailing underscore = pipeline 跑过之后才有)
        self.counts_: Optional[pd.DataFrame] = None
        self.metadata_: Optional[pd.DataFrame] = None
        self.de_results_: Optional[pd.DataFrame] = None
        self.annot_: Optional[pd.DataFrame] = None
        self.mane_map_: Optional[dict] = None
        self.gene_table_: Optional[pd.DataFrame] = None

    def __repr__(self) -> str:
        mark = lambda x: "✓" if x is not None else "·"
        return (
            f"GeneTableBuilder(\n"
            f"  counts       [{mark(self.counts_)}]\n"
            f"  de_results   [{mark(self.de_results_)}]\n"
            f"  annotation   [{mark(self.annot_)}]\n"
            f"  mane_map     [{mark(self.mane_map_)}]\n"
            f"  gene_table   [{mark(self.gene_table_)}]\n"
            f")"
        )
    

    # =========================================================================
    # Step 1: 读 counts
    # =========================================================================
    def load_counts(self) -> pd.DataFrame:
        logger.info("[1/5] Loading counts from %s", self.counts_path)
        full = pd.read_csv(self.counts_path, sep="\t", index_col="GeneID")
 
        samples = self.WT_SAMPLES + self.KO_SAMPLES
        missing = [s for s in samples if s not in full.columns]
        if missing:
            raise ValueError(f"Counts 表中找不到样本: {missing}")
 
        counts = full[samples].astype(int).copy()
        metadata = pd.DataFrame(
            {
                "condition": (
                    ["WT"] * len(self.WT_SAMPLES)
                    + ["METTL3_KO"] * len(self.KO_SAMPLES)
                ),
                "replicate": (
                    list(range(1, len(self.WT_SAMPLES) + 1))
                    + list(range(1, len(self.KO_SAMPLES) + 1))
                ),
            },
            index=samples,
        )

        logger.info("  counts shape: %s", counts.shape)
        logger.info("  总 gene 数 (含 0 表达): %d", len(counts))
        logger.info(
            "  所有样本 count=0 的 gene 数: %d",
            int((counts.sum(axis=1) == 0).sum()),
        )
 
        self.counts_ = counts
        self.metadata_ = metadata
        return counts

    # =========================================================================
    # Step 2: 差异表达分析
    # =========================================================================
    def run_deseq2(self) -> pd.DataFrame:
        if self.counts_ is None:
            self.load_counts()
        logger.info("[2/5] Running DESeq2 via pydeseq2...")
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
        from pydeseq2.default_inference import DefaultInference
 
        inference = DefaultInference(n_cpus=4)
        dds = DeseqDataSet(
            counts=self.counts_.T,
            metadata=self.metadata_,
            design_factors="condition",
            refit_cooks=True,
            inference=inference,
        )
        dds.deseq2()
        stats = DeseqStats(
            dds,
            contrast=["condition", "METTL3_KO", "WT"],
            inference=inference,
        )
        stats.summary()
 
        de = stats.results_df.copy()
        de.index = de.index.astype(int)
        de.index.name = "GeneID"
 
        # 体检
        n_has_lfc = int(de["log2FoldChange"].notna().sum())
        n_sig = int((de["padj"] < 0.05).sum())
        median_lfc = float(de["log2FoldChange"].median())
        logger.info("  DE 结果 shape: %s", de.shape)
        logger.info("  有 log2FC 的 gene 数: %d", n_has_lfc)
        logger.info("  padj < 0.05 的 gene 数: %d (仅参考, 不用于过滤)", n_sig)
        logger.info(
            "  log2FC 中位数: %+.4f (METTL3 KO 预期略 > 0, 因为降解被解除)",
            median_lfc,
        )
 
        self.de_results_ = de
        return de
    
    # =========================================================================
    # Step 3: 读 annotation
    # =========================================================================
    def load_annotation(self) -> pd.DataFrame:
        logger.info("[3/5] Loading annotation from %s", self.annot_path)
 
        annot = pd.read_csv(
            self.annot_path,
            sep="\t",
            dtype={
            "GeneID": int,
            "EnsemblGeneID": str,
            "ChrAcc": str,
            "ChrStart": str,     
            "ChrStop": str,      
            "Orientation": str,
            "Length": str,       
        },
        )
        logger.info("  原始行数: %d", len(annot))
        # 过滤 1: protein-coding
        annot = annot[annot["GeneType"] == "protein-coding"].copy()
        logger.info("  过滤 protein-coding 后: %d", len(annot))

        # 过滤 2: active (去掉 suppressed / discontinued 等)
        annot = annot[annot["Status"] == "active"].copy()
        logger.info("  过滤 active 后: %d", len(annot))

        # --- 处理多坐标 (PAR 基因) ---
        # 策略: 保留第一个坐标 (通常是 chrX), 标记为 is_par.
        # 这样 log2FC 标签不被拆分, PAR 基因作为一个整体训练样本.
        is_multi = annot["ChrAcc"].str.contains(";", na=False)
        n_par = int(is_multi.sum())
        annot["is_par"] = is_multi
        # 拆分并只取第一个
        for col in ["ChrAcc", "ChrStart", "ChrStop", "Orientation", "Length"]:
            annot[col] = annot[col].str.split(";").str[0]

        # 过滤 3: 标准染色体
        annot["chrom"] = annot["ChrAcc"].map(self._refseq_to_chr)
        n_before = len(annot)
        annot = annot[annot["chrom"].notna()].copy()
        logger.info(
            "  过滤标准染色体后: %d (丢弃 %d 个 patch/alt/未定位 contig)",
            len(annot),
            n_before - len(annot),
        )
        annot["ChrStart"] = pd.to_numeric(annot["ChrStart"], errors="coerce").astype("Int64")
        annot["ChrStop"] = pd.to_numeric(annot["ChrStop"], errors="coerce").astype("Int64")
        annot["Length"] = pd.to_numeric(annot["Length"], errors="coerce").astype("Int64")
        n_before = len(annot)
        annot = annot[annot["ChrStart"].notna() & annot["ChrStop"].notna()].copy()
        if n_before > len(annot):
            logger.info(
                "  过滤坐标缺失后: %d (丢弃 %d)",
                len(annot), n_before - len(annot),
            )

        # strand
        strand_map = {"positive": "+", "negative": "-", "+": "+", "-": "-"}
        annot["strand"] = annot["Orientation"].map(strand_map)
        
        # 改名 + 去 Ensembl 版本号
        annot = annot.rename(
            columns={
                "GeneID": "gene_id_entrez",
                "Symbol": "symbol",
                "EnsemblGeneID": "gene_id_ensembl",
                "ChrStart": "start",
                "ChrStop": "end",
                "Length": "gene_length",
                "GeneType": "biotype",
            }
        )
        annot["gene_id_ensembl"] = annot["gene_id_ensembl"].map(
            self._strip_ensembl_version
        )

        annot["start"] = annot["start"].astype("int64")
        annot["end"] = annot["end"].astype("int64")
        annot["gene_length"] = annot["gene_length"].astype("int64")


        cols = [
            "gene_id_entrez",
            "gene_id_ensembl",
            "symbol",
            "chrom",
            "start",
            "end",
            "strand",
            "gene_length",
            "biotype",
            "is_par",    
        ]
        annot = annot[cols].set_index("gene_id_entrez")
        logger.info("  最终 protein-coding gene 数: %d", len(annot))
        logger.info(
            "  有 Ensembl ID 的比例: %.1f%%",
            100 * annot["gene_id_ensembl"].notna().mean(),
        )
        if annot["is_par"].any():
            logger.info("  PAR 基因 (X/Y 双拷贝): %d", int(annot["is_par"].sum()))
 
        self.annot_ = annot
        return annot
    
    # =========================================================================
    # Step 4: 从 GTF 提取 MANE Select
    # =========================================================================
    def load_mane_transcripts(self) -> dict:
        logger.info("[4/5] Loading MANE Select from %s", self.gtf_path)
        mane_map: dict = {}
        n_transcript_lines = 0
        n_mane = 0
 
        opener = self._open_maybe_gz(self.gtf_path)
        with opener as f:
            for line in f:
                if line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 9 or fields[2] != "transcript":
                    continue
                n_transcript_lines += 1
 
                attrs = fields[8]
                if "MANE_Select" not in attrs:
                    continue
 
                gene_id = self._extract_gtf_attr(attrs, "gene_id")
                tx_id = self._extract_gtf_attr(attrs, "transcript_id")
                if gene_id is None or tx_id is None:
                    continue
 
                gene_id = self._strip_ensembl_version(gene_id)
                tx_id = self._strip_ensembl_version(tx_id)
 
                mane_map[gene_id] = tx_id
                n_mane += 1
 
        logger.info("  扫描 transcript 行数: %d", n_transcript_lines)
        logger.info("  找到 MANE Select 行: %d", n_mane)
        logger.info("  unique genes with MANE: %d", len(mane_map))
 
        self.mane_map_ = mane_map
        return mane_map
    
    # =========================================================================
    # Step 5: 构建最终表
    # =========================================================================
    def build_table(self) -> pd.DataFrame:
        if self.annot_ is None:
            self.load_annotation()
        if self.de_results_ is None:
            self.run_deseq2()
        if self.mane_map_ is None:
            self.load_mane_transcripts()
 
        logger.info("[5/5] Building gene_table...")
 
        table = self.annot_.copy()
        # --- Join DE (按 index = gene_id_entrez) ---
        de = self.de_results_.rename(
            columns={
                "baseMean": "basemean",
                "log2FoldChange": "log2fc",
                "lfcSE": "lfc_se",
                "pvalue": "pvalue",
                "padj": "padj",
            }
        )[["basemean", "log2fc", "lfc_se", "pvalue", "padj"]]
        table = table.join(de, how="left")

        # --- Join MANE (按 gene_id_ensembl) ---
        # 用 map 代替 merge, 保留原 index, 避免 reset_index 麻烦
        table["mane_transcript_id"] = table["gene_id_ensembl"].map(self.mane_map_)
 
        # --- 预留列 (后续 m6A / reader 数据填充) ---
        # 用可空整型 Int64, 区分"未计算"(NA) 和"计算过但是 0".
        table["n_m6a_sites"] = pd.array([pd.NA] * len(table), dtype="Int64")
        table["n_ythdf_sites"] = pd.array([pd.NA] * len(table), dtype="Int64")
        table["n_igf2bp_sites"] = pd.array([pd.NA] * len(table), dtype="Int64")

        col_order = [
            # 身份
            "gene_id_ensembl", "symbol",
            # 坐标
            "chrom", "start", "end", "strand", "gene_length", "biotype", "is_par", 
            # 代表性 transcript
            "mane_transcript_id",
            # 训练标签 (DE)
            "log2fc", "padj", "basemean", "lfc_se", "pvalue",
            # 预留: m6A / reader 统计
            "n_m6a_sites", "n_ythdf_sites", "n_igf2bp_sites",
        ]
        table = table[col_order]
 
        self._print_build_stats(table)
 
        self.gene_table_ = table
        return table
    
    def _print_build_stats(self, table: pd.DataFrame) -> None:
    
        n = len(table)
        has_ensg = table["gene_id_ensembl"].notna()
        has_lfc = table["log2fc"].notna()
        has_mane = table["mane_transcript_id"].notna()
        all_three = has_ensg & has_lfc & has_mane

        # baseMean 过滤统计(表达量阈值)
        basemean_ge_10 = table["basemean"] >= 10
        basemean_ge_50 = table["basemean"] >= 50
        trainable_bm10 = all_three & basemean_ge_10
        trainable_bm50 = all_three & basemean_ge_50

        logger.info("=" * 60)
        logger.info("gene_table 构建完成: %d protein-coding genes", n)
        logger.info(
            "  有 Ensembl ID:           %6d (%.1f%%)",
            int(has_ensg.sum()), 100 * has_ensg.mean(),
        )
        logger.info(
            "  有 log2FC (DE 结果):     %6d (%.1f%%)",
            int(has_lfc.sum()), 100 * has_lfc.mean(),
        )
        logger.info(
            "  有 MANE transcript:      %6d (%.1f%%)",
            int(has_mane.sum()), 100 * has_mane.mean(),
        )
        logger.info(
            "  三者都有 (训练样本上限):   %6d (%.1f%%)",
            int(all_three.sum()), 100 * all_three.mean(),
        )
        logger.info("-" * 60)
        logger.info("  表达量过滤 (用于训练时筛选):")
        logger.info(
            "    + baseMean >= 10:      %6d (%.1f%%)  [宽松, 保留更多样本]",
            int(trainable_bm10.sum()), 100 * trainable_bm10.mean(),
        )
        logger.info(
            "    + baseMean >= 50:      %6d (%.1f%%)  [严格, 标签更可信]",
            int(trainable_bm50.sum()), 100 * trainable_bm50.mean(),
        )
        logger.info("  ^ 训练样本需要 >1000 才够训练深度模型 (见方案风险一)")
        logger.info("=" * 60)

    def save(self, path=None) -> Path:
        if self.gene_table_ is None:
            raise RuntimeError("必须先调用 build_table() 或 run_all()")
 
        out = Path(path) if path else self.output_path
        if out is None:
            raise ValueError("未指定输出路径")
 
        out.parent.mkdir(parents=True, exist_ok=True)
        self.gene_table_.to_parquet(out, engine="pyarrow", compression="snappy")
        logger.info("Saved gene_table to %s", out)
        return out

    def run_all(self) -> pd.DataFrame:
        
        self.load_counts()
        self.run_deseq2()
        self.load_annotation()
        self.load_mane_transcripts()
        self.build_table()
        if self.output_path is not None:
            self.save()
        return self.gene_table_
    

    @classmethod
    def _refseq_to_chr(cls, acc):
        """NC_000001.11 -> chr1; 非标准染色体返回 None."""
        if not isinstance(acc, str):
            return None
        prefix = acc.split(".")[0]
        return cls._REFSEQ_CHROM_MAP.get(prefix)
 
    @staticmethod
    def _strip_ensembl_version(eid):
        """ENSG00000186092.12 -> ENSG00000186092; 空值返回 None."""
        if not isinstance(eid, str) or not eid:
            return None
        return eid.split(".")[0]
 
    @classmethod
    def _extract_gtf_attr(cls, attrs: str, key: str):
        """从 GTF attributes 字段提取 `key "value"` 中的 value."""
        if key not in cls._ATTR_RE_CACHE:
            cls._ATTR_RE_CACHE[key] = re.compile(rf'{key} "([^"]+)"')
        m = cls._ATTR_RE_CACHE[key].search(attrs)
        return m.group(1) if m else None
 
    @staticmethod
    def _open_maybe_gz(path: Path):
        """根据文件名后缀自动选择 gzip 或普通打开."""
        if str(path).endswith(".gz"):
            return gzip.open(path, "rt")
        return open(path, "rt")
 

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
 
    ROOT = Path("/media/sda5/xsh-workspace/m6A_MIL")
 
    builder = GeneTableBuilder(
        counts_path=ROOT / "data/raw/RNA-seq/GSE182607_raw_counts_GRCh38.p13_NCBI.tsv",
        annot_path=ROOT / "data/raw/RNA-seq/Human.GRCh38.p13.annot.tsv",
        gtf_path=ROOT / "data/raw/Homo_sapiens.GRCh38.115.chr.gtf.gz", 
        output_path=ROOT / "data/processed/gene_table.parquet",
    )
 
    builder.run_all()

    # missing = builder.annot_[builder.annot_['gene_id_ensembl'].isna()]
    # print(f"缺失 Ensembl ID 的基因数: {len(missing)}")
    # print(missing[['symbol', 'chrom', 'gene_length']].head(20))

    # import matplotlib.pyplot as plt
    # import numpy as np
    # df = builder.de_results_.dropna(subset=['log2FoldChange'])
    # log2fc = df['log2FoldChange']

    # fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # # 子图 1: 全范围分布
    # axes[0].hist(log2fc, bins=100, color='steelblue', edgecolor='white', alpha=0.8)
    # axes[0].axvline(0, color='red', linestyle='--', linewidth=1)
    # axes[0].axvline(log2fc.median(), color='orange', linestyle='--', linewidth=1,
    #                 label=f'median={log2fc.median():.3f}')
    # axes[0].set_xlabel('log2FC')
    # axes[0].set_ylabel('Number of genes')
    # axes[0].set_title(f'Full distribution (n={len(log2fc)})')
    # axes[0].legend()

    # # 子图 2: 中间段放大 (-3 到 +3)
    # center = log2fc[(log2fc > -3) & (log2fc < 3)]
    # axes[1].hist(center, bins=100, color='steelblue', edgecolor='white', alpha=0.8)
    # axes[1].axvline(0, color='red', linestyle='--', linewidth=1)
    # axes[1].set_xlabel('log2FC')
    # axes[1].set_ylabel('Number of genes')
    # axes[1].set_title(f'Zoomed: -3 to +3 (n={len(center)})')

    # # 子图 3: MA plot (baseMean vs log2FC, 经典 QC 图)
    # significant = df['padj'] < 0.05
    # axes[2].scatter(df.loc[~significant, 'baseMean'], df.loc[~significant, 'log2FoldChange'],
    #                 s=2, color='gray', alpha=0.3, label='padj ≥ 0.05')
    # axes[2].scatter(df.loc[significant, 'baseMean'], df.loc[significant, 'log2FoldChange'],
    #                 s=2, color='red', alpha=0.6, label=f'padj < 0.05 (n={significant.sum()})')
    # axes[2].axhline(0, color='black', linewidth=0.5)
    # axes[2].set_xscale('log')
    # axes[2].set_xlabel('baseMean (log scale)')
    # axes[2].set_ylabel('log2FC')
    # axes[2].set_title('MA plot')
    # axes[2].legend(markerscale=3)

    # plt.tight_layout()
    # plt.savefig(f"{ROOT}/de_qc.png", dpi=150)
    # plt.show()
    print(builder)
