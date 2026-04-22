
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
import json
import logging
import random
import time
from functools import partial
from pathlib import Path
 
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
 
from src.data.dataset import M6ADataset
from src.data.collate import m6a_collate_fn, InstanceBudgetSampler
from src.data.encoding import SequenceEncoder
from src.model.m6a_mil_model import M6AMIL
from src.training.trainer import Trainer
from src.training.losses import WeightedMSELoss
from src.training.metrics import compute_metrics
from src.utils.config import load_config, save_config
 
logger = logging.getLogger(__name__)
 

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d", seed)
 
 
def load_split(split_dir: str, fold: int) -> dict:
    split_path = Path(split_dir) / f"fold_{fold}.json"
    with open(split_path) as f:
        split = json.load(f)
    logger.info(
        "Fold %d loaded: train=%d, val=%d, test=%d genes",
        fold,
        len(split["train_gene_ids"]),
        len(split["val_gene_ids"]),
        len(split["test_gene_ids"]),
    )
    return split
 

def setup_experiment(config, fold: int) -> Path:
   
    exp_name = config["experiment"]["name"]
    exp_dir = Path(config["experiment"]["output_dir"]) / f"{exp_name}_fold{fold}"
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "results").mkdir(parents=True, exist_ok=True)
 
    save_config(config, exp_dir / "config.yaml")
    logger.info("Experiment dir: %s", exp_dir)
    return exp_dir

def build_dataloaders(split, gene_table, site_table, config):
    encoder_method = config["model"]["site_encoder"]["type"]
    encoder = SequenceEncoder(
        method=encoder_method,
        seq_len=config["data"]["seq_len"],
    )
    logger.info("Sequence encoder: %s", encoder_method)
 
    collate = partial(m6a_collate_fn, encoder=encoder)

    # --- Dataset ---
    train_ds = M6ADataset(
        gene_ids=split["train_gene_ids"],
        gene_table=gene_table,
        site_table=site_table,
        seq_dir=config["data"]["seq_dir"],
        config=config["data"],
    )
    val_ds = M6ADataset(
        gene_ids=split["val_gene_ids"],
        gene_table=gene_table,
        site_table=site_table,
        seq_dir=config["data"]["seq_dir"],
        config=config["data"],
    )
    test_ds = M6ADataset(
        gene_ids=split["test_gene_ids"],
        gene_table=gene_table,
        site_table=site_table,
        seq_dir=config["data"]["seq_dir"],
        config=config["data"],
    )
    
    # --- Instance Budget Sampler (train) ---
    budget = config["training"]["instance_budget"]
    max_bs = config["training"]["max_batch_size"]
    train_sampler = InstanceBudgetSampler(
        n_sites_per_gene=train_ds.n_sites_per_gene,
        instance_budget=budget,
        max_batch_size=max_bs,
        shuffle=True,
        min_batch_size=1,
        seed=config["training"]["seed"],
    )
    stats = train_sampler.batch_stats()
    logger.info("Train sampler: %s", stats)
 
    val_sampler = InstanceBudgetSampler(
        n_sites_per_gene=val_ds.n_sites_per_gene,
        instance_budget=budget,
        max_batch_size=max_bs,
        min_batch_size=1,
        shuffle=False,
    )
    test_sampler = InstanceBudgetSampler(
        n_sites_per_gene=test_ds.n_sites_per_gene,
        instance_budget=budget,
        max_batch_size=max_bs,
        min_batch_size=1,
        shuffle=False,
    )


    # --- DataLoader ---
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=collate,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        collate_fn=collate,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_sampler=test_sampler,
        collate_fn=collate,
        num_workers=0,
        pin_memory=True,
    )
 
    logger.info(
        "DataLoaders built: train=%d batches, val=%d batches, test=%d batches",
        len(train_loader), len(val_loader), len(test_loader),
    )
    return train_loader, val_loader, test_loader


def train_one_fold(fold: int, config: dict):
    logger.info("=" * 60)
    logger.info("Training fold %d / %d", fold, 5)
    logger.info("=" * 60)
 
    # --- 0. 设置 ---
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    exp_dir = setup_experiment(config, fold)    
    file_handler = logging.FileHandler(exp_dir / "training.log")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logging.getLogger().addHandler(file_handler)

    # --- 1. 加载数据表 ---
    logger.info("Loading data tables...")
    gene_table = pd.read_parquet(config["data"]["gene_table"])
    if gene_table.index.name == "gene_id_entrez":
        gene_table = gene_table.reset_index()
    site_table = pd.read_parquet(config["data"]["site_table"])
    logger.info("  gene_table: %d rows", len(gene_table))
    logger.info("  site_table: %d rows", len(site_table))

    # --- 2. 加载 split ---
    split = load_split(config["data"]["split_dir"], fold)

    # --- 3. 构建 DataLoader ---
    train_loader, val_loader, test_loader = build_dataloaders(
        split, gene_table, site_table, config,
    )

    # --- 4. 构建模型 ---
    model = M6AMIL(config["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s", model.__class__.__name__)
    logger.info("  Trainable parameters: %d (%.2f M)", n_params, n_params / 1e6)

    # --- 5. 构建训练器 ---
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(exp_dir / "logs"))
    # criterion = WeightedMSELoss(use_weights=False)
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=config["training"],
        device=device,
        exp_dir=exp_dir,
        writer=writer,   
    )   

    # --- 6. 训练 ---
    logger.info("-" * 60)
    logger.info("Start training...")
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    logger.info("Training finished in %.1f min", elapsed / 60)

    # --- 7. 加载最佳模型, 在 test 集上评估 ---
    logger.info("-" * 60)
    logger.info("Evaluating on test set...")
    best_ckpt = exp_dir / "checkpoints" / "best.pt"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        logger.info("  Loaded best checkpoint: %s", best_ckpt)
    else:
        logger.warning("  No best checkpoint found, using last model state")
 
    test_preds, test_labels, test_attn, test_gene_ids = trainer.predict(test_loader)
    test_metrics = compute_metrics(test_preds, test_labels)
 
    logger.info("Test metrics:")
    for k, v in test_metrics.items():
        logger.info("  %-15s: %.4f", k, v)
    
    # --- 8. 保存结果 ---
    results = {
        "fold": fold,
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "n_train_genes": len(split["train_gene_ids"]),
        "n_val_genes": len(split["val_gene_ids"]),
        "n_test_genes": len(split["test_gene_ids"]),
        "training_time_min": round(elapsed / 60, 1),
    }
    results_path = exp_dir / "results" / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    # 保存逐基因预测结果 (用于后续生物学分析)
    pred_df = pd.DataFrame({
        "gene_id_entrez": test_gene_ids,
        "log2fc_true": test_labels,
        "log2fc_pred": test_preds,
    })
    pred_path = exp_dir / "results" / "test_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    logger.info("Predictions saved to %s", pred_path)
    writer.close()
    return test_metrics

def main():
    parser = argparse.ArgumentParser(description="m6A MIL Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="配置文件路径")
    parser.add_argument("--fold", type=str, default="1",
                        help="1-5 or 'all'")
    # 常用参数覆盖 (优先于 config 文件)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()
 
    # --- 加载配置 ---
    config = load_config(args.config)

    # --- 命令行参数覆盖 ---
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.exp_name is not None:
        config["experiment"]["name"] = args.exp_name

    # --- 设置日志 ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- 训练 ---
    if args.fold.lower() == "all":
        all_metrics = {}
        for fold in range(1, 6):
            metrics = train_one_fold(fold, config)
            all_metrics[f"fold_{fold}"] = metrics

        logger.info("=" * 60)
        logger.info("5-Fold Cross-Validation Summary:")
        metric_names = list(next(iter(all_metrics.values())).keys())
        for metric in metric_names:
            values = [all_metrics[f][metric] for f in all_metrics]
            mean = np.mean(values)
            std = np.std(values)
            logger.info("  %-15s: %.4f ± %.4f", metric, mean, std)

        summary_dir = Path(config["experiment"]["output_dir"]) / config["experiment"]["name"]
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "folds": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_metrics.items()},
            "mean": {m: float(np.mean([all_metrics[f][m] for f in all_metrics])) for m in metric_names},
            "std": {m: float(np.std([all_metrics[f][m] for f in all_metrics])) for m in metric_names},
        }
        with open(summary_dir / "cv_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("CV summary saved to %s", summary_dir / "cv_summary.json")
    
    else:
        # 单折
        fold = int(args.fold)
        train_one_fold(fold, config)
    

if __name__ == "__main__":
    main()
