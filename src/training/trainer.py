from __future__ import annotations
 
import json
import logging
import time
from pathlib import Path
 
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
 
from src.training.losses import WeightedMSELoss, compute_sample_weights
from src.training.metrics import compute_metrics
 
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion: nn.Module,
        config: dict,
        device: torch.device,
        exp_dir: Path,
        writer=None,

    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        self.device = device
        self.exp_dir = exp_dir
        self.writer = writer 

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.best_val_loss = float("inf")
        self.best_val_metrics = {}
        self.patience_counter = 0
        self.patience = config.get("patience", 15)

        self.current_epoch = 0
        self.train_history = []
        self.val_history = []

    def _build_optimizer(self) -> AdamW:
        lr_encoder = self.config.get("lr_encoder", 2e-5)
        lr_head = self.config.get("lr_head", 1e-3)
        weight_decay = self.config.get("weight_decay", 1e-4)

        param_groups = self.model.get_parameter_groups(lr_encoder, lr_head)
        optimizer = AdamW(param_groups, weight_decay=weight_decay)
 
        n_encoder = sum(p.numel() for p in param_groups[0]["params"])
        n_head = sum(p.numel() for p in param_groups[1]["params"])
        logger.info(
            "Optimizer: AdamW, encoder lr=%.1e (%d params), head lr=%.1e (%d params), wd=%.1e",
            lr_encoder, n_encoder, lr_head, n_head, weight_decay,
        )
        return optimizer
    
    def _build_scheduler(self):
        """Cosine annealing with linear warmup."""
        epochs = self.config.get("epochs", 50)
        warmup_epochs = self.config.get("warmup_epochs", 5)
 
        warmup = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=1e-5,
        )
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
        logger.info(
            "Scheduler: linear warmup (%d epochs) + cosine annealing (%d epochs)",
            warmup_epochs, epochs - warmup_epochs,
        )
        return scheduler
    
    def train(self):
        epochs = self.config.get("epochs", 50)
        grad_clip = self.config.get("gradient_clip", 1.0)

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            train_loss = self._train_one_epoch(grad_clip)
            val_loss, val_metrics = self._validate()

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.train_history.append(train_loss)
            self.val_history.append(val_loss)
 
            elapsed = time.time() - epoch_start

            logger.info(
                "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  "
                "pearson=%.4f  spearman=%.4f  dir_acc=%.3f  "
                "lr=%.1e  time=%.0fs",
                epoch, epochs, train_loss, val_loss,
                val_metrics.get("pearson_r", 0),
                val_metrics.get("spearman_r", 0),
                val_metrics.get("direction_acc", 0),
                current_lr, elapsed,
            )
            # --- TensorBoard ---
            if self.writer is not None:
                self.writer.add_scalars("loss", {
                    "train": train_loss,
                    "val": val_loss,
                }, epoch)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)
                self.writer.add_scalar("lr/encoder", self.optimizer.param_groups[0]["lr"], epoch)
                self.writer.add_scalar("lr/head", self.optimizer.param_groups[1]["lr"], epoch)
        
            # --- Checkpoint ---
            self._save_checkpoint("last.pt")
 
            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.best_val_metrics = val_metrics
                self.patience_counter = 0
                self._save_checkpoint("best.pt")
                logger.info(
                    "  ↑ New best (val_loss improved by %.4f)", improvement,
                )
            else:
                self.patience_counter += 1
                logger.info(
                    "  patience: %d/%d", self.patience_counter, self.patience,
                )
 
            # --- Early stopping ---
            if self.patience_counter >= self.patience:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

        self._save_history()
        logger.info(
            "Training complete. Best val_loss=%.4f at epoch %d",
            self.best_val_loss,
            self.train_history.index(min(self.train_history)) + 1
            if self.train_history else 0,
        )
        logger.info("Best val metrics: %s", self.best_val_metrics)
 

    def _train_one_epoch(self, grad_clip: float) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        log_interval = 20
        for step, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)
            output = self.model(
                input_ids=batch["input_ids"],
                token_attn_mask=batch["token_attn_mask"],
                scalars=batch["scalars"],
                site_mask=batch["site_mask"],
            )

            sample_weights = None
            if "basemean" in batch:
                sample_weights = compute_sample_weights(batch["basemean"])

            loss = self.criterion(
                output["predictions"],
                batch["labels"],
                sample_weights=sample_weights,
            )

            self.optimizer.zero_grad()
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=grad_clip,
                )
            self.optimizer.step()
 
            total_loss += loss.item()
            n_batches += 1
            if (step + 1) % log_interval == 0:
                avg_loss = total_loss / n_batches
                logger.info(
                    "  [batch %d/%d]  loss=%.4f  sites=%d",
                    step + 1, len(self.train_loader),
                    avg_loss,
                    sum(batch["n_sites"]),
                )
 
        return total_loss / max(n_batches, 1)
            
    @torch.no_grad()
    def _validate(self) -> tuple[float, dict]:
        """验证, 返回 (平均 loss, metrics dict)."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_labels = []

        for batch in self.val_loader:
            batch = self._to_device(batch)
 
            output = self.model(
                input_ids=batch["input_ids"],
                token_attn_mask=batch["token_attn_mask"],
                scalars=batch["scalars"],
                site_mask=batch["site_mask"],
            )

            loss = self.criterion(
                output["predictions"],
                batch["labels"],
                sample_weights=None,
            )
 
            total_loss += loss.item()
            n_batches += 1
 
            all_preds.append(output["predictions"].squeeze(-1).cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
 
        avg_loss = total_loss / max(n_batches, 1)
 
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        metrics = compute_metrics(preds, labels)
 
        return avg_loss, metrics
 
    @torch.no_grad()
    def predict(self, dataloader) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        self.model.eval()
        all_preds = []
        all_labels = []
        all_attn = []
        all_gene_ids = []
 
        for batch in dataloader:
            batch = self._to_device(batch)
 
            output = self.model(
                input_ids=batch["input_ids"],
                token_attn_mask=batch["token_attn_mask"],
                scalars=batch["scalars"],
                site_mask=batch["site_mask"],
            )
 
            preds = output["predictions"].squeeze(-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            attn = output["attention"].cpu().numpy()        # (B, S)
            n_sites = batch["n_sites"]                       # list[int]
 
            all_preds.append(preds)
            all_labels.append(labels)
            all_gene_ids.extend(batch["gene_ids"])
 
            # attention: 去掉 padding 部分, 只保留真实位点
            for i in range(len(n_sites)):
                n = n_sites[i]
                all_attn.append(attn[i, :n])
 
        predictions = np.concatenate(all_preds)
        labels_arr = np.concatenate(all_labels)
 
        return predictions, labels_arr, all_attn, all_gene_ids
    
    def _to_device(self, batch: dict) -> dict:
        """将 batch 中的 tensor 移动到 GPU."""
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device, non_blocking=True)
            else:
                result[k] = v
        return result
 
    def _save_checkpoint(self, name: str):
        """保存模型 checkpoint."""
        ckpt_path = self.exp_dir / "checkpoints" / name
        torch.save(self.model.state_dict(), ckpt_path)
 
    def _save_history(self):
        """保存训练历史."""
        history = {
            "train_loss": self.train_history,
            "val_loss": self.val_history,
            "best_val_loss": self.best_val_loss,
            "best_val_metrics": {
                k: float(v) for k, v in self.best_val_metrics.items()
            },
            "stopped_epoch": self.current_epoch,
        }
        history_path = self.exp_dir / "results" / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
 