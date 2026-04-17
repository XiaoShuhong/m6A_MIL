





 
from __future__ import annotations
 
import logging
import subprocess
from pathlib import Path
 
import h5py
import pandas as pd
 
logger = logging.getLogger(__name__)
FLANK = 250           # 两侧各取 250nt
SEQ_LEN = 2 * FLANK + 1  # 501nt

COMPLEMENT = {
    'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
    'a': 't', 't': 'a', 'c': 'g', 'g': 'c',
    'N': 'N', 'n': 'n',
}

def reverse_complement(seq: str) -> str:
    """反向互补."""
    return ''.join(COMPLEMENT.get(b, 'N') for b in reversed(seq))
 
def make_site_id(chrom: str, pos: int, strand: str) -> str:
    """生成 site_id: chr1_12345_+"""
    return f"{chrom}_{pos}_{strand}"
 

def decompress_fasta_if_needed(gz_path: Path) -> Path:
    """
    如果 FASTA 是 .gz 格式, 解压并返回解压后的路径.
    pyfaidx 可以处理 bgzf 格式的 .gz, 但普通 gzip 不行,
    所以统一解压更安全.
    """
    if not str(gz_path).endswith('.gz'):
        return gz_path
    
    decompressed = gz_path.with_suffix('')  # 去掉 .gz
    if decompressed.exists():
        logger.info("  已存在解压文件: %s", decompressed)
        return decompressed
    
    logger.info("  解压基因组 FASTA (首次运行, ~2 分钟)...")
    logger.info("  %s → %s", gz_path, decompressed)
    subprocess.run(['gunzip', '-k', str(gz_path)], check=True)
    logger.info("  解压完成")
    return decompressed


def extract_sequences(
    site_table_path: str | Path,
    genome_fasta_path: str | Path,
    output_dir: str | Path,
    flank: int = FLANK,
    chunk_size: int = 10000,
):
    import pyfaidx
 
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_len = 2 * flank + 1

    # =========================================================================
    # 1. 读 site_table
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Loading site_table...")
    st = pd.read_parquet(site_table_path)
    logger.info("  总行数: %d", len(st))

    # 去重: 同一个 (chrom, pos, strand) 可能映射到多个基因
    # 序列只需提取一次
    unique_sites = (
        st[["chrom", "pos", "strand"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    logger.info("  唯一位点数: %d", len(unique_sites))

    # =========================================================================
    # 2. 加载基因组 FASTA
    # =========================================================================
    logger.info("-" * 60)
    genome_path = Path(genome_fasta_path)

    # 解压 (如果是 .gz)
    genome_path = decompress_fasta_if_needed(genome_path)
    
    logger.info("Loading genome FASTA: %s", genome_path)
    logger.info("  (首次运行会建索引, 需要 ~2 分钟)")
    genome = pyfaidx.Fasta(str(genome_path), as_raw=True)
    
    # 确认染色体命名
    fasta_chroms = set(genome.keys())
    sample_chroms = list(fasta_chroms)[:5]
    logger.info("  FASTA 染色体命名示例: %s", sample_chroms)

    # 判断命名风格: "1" vs "chr1"
    has_chr_prefix = any(c.startswith("chr") for c in fasta_chroms)
    if has_chr_prefix:
        chrom_transform = lambda c: c  # site_table 和 FASTA 都是 chr1 风格
        logger.info("  染色体命名: chr1 风格 (和 site_table 一致)")
    else:
        chrom_transform = lambda c: c.replace("chr", "")  # site_table 是 chr1, FASTA 是 1
        logger.info("  染色体命名: 数字风格 (1, 2, ...) → 需要转换")

    # =========================================================================
    # 3. 按染色体分组提取
    # =========================================================================
    logger.info("=" * 60)
    logger.info("开始提取序列 (±%dnt = %dnt)...", flank, seq_len)
 
    chrom_groups = unique_sites.groupby("chrom")
    n_chroms = len(chrom_groups)
    
    total_extracted = 0
    total_skipped = 0
    total_n_heavy = 0  # N 碱基占比 > 50% 的位点
    h5_files = {}

    for chrom_idx, (chrom, group) in enumerate(chrom_groups, 1):
        fasta_chrom = chrom_transform(chrom)
        # 检查染色体是否存在于 FASTA
        if fasta_chrom not in genome:
            logger.warning(
                "  ⚠ 染色体 %s (FASTA key: %s) 不在基因组中, 跳过 %d 个位点",
                chrom, fasta_chrom, len(group),
            )
            total_skipped += len(group)
            continue

        chrom_seq = genome[fasta_chrom]
        chrom_len = len(chrom_seq)
        # 创建 HDF5 文件
        h5_path = output_dir / f"{chrom}.h5"
        h5 = h5py.File(str(h5_path), "w")
        h5_files[chrom] = h5_path
        
        n_extracted = 0
        n_skipped = 0
        n_n_heavy = 0
        for _, row in group.iterrows():
            pos = int(row["pos"])       # 1-based (GLORI 坐标)
            strand = row["strand"]
            site_id = make_site_id(chrom, pos, strand)
            
            # 转 0-based: 中心位置 = pos - 1
            center_0 = pos - 1
            start_0 = center_0 - flank
            end_0 = center_0 + flank + 1  # exclusive
            
            # 边界处理: 用 N 填充
            left_pad = 0
            right_pad = 0
            actual_start = start_0
            actual_end = end_0
            
            if start_0 < 0:
                left_pad = -start_0
                actual_start = 0
            if end_0 > chrom_len:
                right_pad = end_0 - chrom_len
                actual_end = chrom_len
            
            # 提取序列
            raw_seq = str(chrom_seq[actual_start:actual_end])
            
            # 填充边界
            if left_pad > 0 or right_pad > 0:
                raw_seq = 'N' * left_pad + raw_seq + 'N' * right_pad
            
            # 长度校验
            if len(raw_seq) != seq_len:
                logger.warning(
                    "  ⚠ %s: 序列长度 %d ≠ %d, 跳过",
                    site_id, len(raw_seq), seq_len,
                )
                n_skipped += 1
                continue
            # 负链: 反向互补
            if strand == '-':
                raw_seq = reverse_complement(raw_seq)
            
            # N 碱基统计
            n_count = raw_seq.upper().count('N')
            if n_count > seq_len * 0.5:
                n_n_heavy += 1
            
            # 统一大写
            raw_seq = raw_seq.upper()
             
            # 写入 HDF5 (原始字符串)
            h5.create_dataset(site_id, data=raw_seq)
            n_extracted += 1
        h5.close()
        total_extracted += n_extracted
        total_skipped += n_skipped
        total_n_heavy += n_n_heavy
        logger.info(
            "  [%2d/%d] %5s: %5d extracted, %d skipped, %d N-heavy → %s",
            chrom_idx, n_chroms, chrom,
            n_extracted, n_skipped, n_n_heavy,
            h5_path.name,
        )
    # =========================================================================
    # 4. 汇总统计
    # =========================================================================
    logger.info("=" * 60)
    logger.info("序列提取完成:")
    logger.info("  总提取: %d", total_extracted)
    logger.info("  总跳过: %d", total_skipped)
    logger.info("  N 碱基 >50%%: %d (%.2f%%)",
                total_n_heavy,
                100 * total_n_heavy / max(total_extracted, 1))
    logger.info("  序列长度: %d nt (±%d)", seq_len, flank)
    logger.info("  存储格式: 原始 ATCGN 字符串 (训练时按需编码)")
    logger.info("  输出目录: %s", output_dir)
    logger.info("  HDF5 文件数: %d", len(h5_files))
    
    # 打印文件大小
    total_size = 0
    for chrom, path in sorted(h5_files.items(), key=lambda x: x[0]):
        size_mb = path.stat().st_size / 1024 / 1024
        total_size += size_mb
    logger.info("  总大小: %.1f MB", total_size)
    # =========================================================================
    # 5. 验证: 随机抽几个位点检查
    # =========================================================================
    logger.info("-" * 60)
    logger.info("验证 (随机抽 3 个位点):")
    
    sample_sites = unique_sites.sample(min(3, len(unique_sites)), random_state=42)
    for _, row in sample_sites.iterrows():
        chrom = row["chrom"]
        pos = int(row["pos"])
        strand = row["strand"]
        site_id = make_site_id(chrom, pos, strand)
        
        h5_path = output_dir / f"{chrom}.h5"
        if not h5_path.exists():
            continue
            
        with h5py.File(str(h5_path), "r") as h5:
            if site_id not in h5:
                logger.warning("  ⚠ %s 未找到!", site_id)
                continue
            
            # 读取字符串
            seq_str = h5[site_id][()].decode() if isinstance(h5[site_id][()], bytes) else str(h5[site_id][()])
            
            # 显示中心 ±5nt
            center = len(seq_str) // 2
            context = seq_str[center-5:center+6]
            center_base = seq_str[center]
            
            logger.info(
                "  %s: len=%d, center=%s, context=...%s..., N_count=%d",
                site_id, len(seq_str), center_base, context,
                seq_str.count('N'),
            )
            # m6A 位点的中心碱基应该是 A (正链) 或已经反向互补后的 A
            if center_base != 'A':
                logger.warning(
                    "    ⚠ 中心碱基不是 A (是 %s)! 检查坐标/链方向",
                    center_base,
                )
 
    logger.info("=" * 60)
    logger.info("Done!")
 
 

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
 
    ROOT = Path("/media/sda5/xsh-workspace/m6A_MIL")
 
    extract_sequences(
        site_table_path=ROOT / "data/processed/site_table.parquet",
        genome_fasta_path=ROOT / "data/raw/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz",
        output_dir=ROOT / "data/processed/sequences",
        flank=250,
    )
