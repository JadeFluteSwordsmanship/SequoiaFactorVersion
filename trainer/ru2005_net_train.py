
from __future__ import annotations
import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import copy
import numpy as np

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from scipy.stats import spearmanr, pearsonr
from datetime import datetime
import uuid


try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int = 1223):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_corrs(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out = {"pearson": np.nan, "spearman": np.nan}
    try:
        if np.std(y_true) > 0 and np.std(y_pred) > 0:
            out["pearson"] = float(pearsonr(y_true, y_pred)[0])
    except Exception:
        pass
    try:
        out["spearman"] = float(spearmanr(y_true, y_pred).statistic)
    except Exception:
        pass
    return out


def topk_hit_rate(y_true: np.ndarray, y_score: np.ndarray, k: float = 0.05) -> Tuple[float, float]:
    """
    Top-k 命中率：预测最高 k 部分里，y>0 的占比；
    Bottom-k 命中率：预测最低 k 部分里，y<0 的占比。
    """
    n = len(y_true)
    k_n = max(1, int(round(n * k)))
    order = np.argsort(y_score)  
    bottom_idx = order[:k_n]
    top_idx = order[-k_n:]
    top_rate = float(np.mean(y_true[top_idx] > 0))
    bottom_rate = float(np.mean(y_true[bottom_idx] < 0))
    return top_rate, bottom_rate


# ============================================================
# Datasets
# ============================================================

class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = X.astype(np.float32)
        self.y = None if y is None else y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        xi = self.X[idx]
        if self.y is None:
            return torch.from_numpy(xi)
        yi = self.y[idx]
        return torch.from_numpy(xi), torch.tensor(yi, dtype=torch.float32)


class SeqDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y: Optional[np.ndarray] = None):
        """
        X_seq: (N, L, F)
        """
        self.X = X_seq.astype(np.float32)
        self.y = None if y is None else y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        xi = self.X[idx]  
        if self.y is None:
            return torch.from_numpy(xi)
        yi = self.y[idx]
        return torch.from_numpy(xi), torch.tensor(yi, dtype=torch.float32)


# ============================================================
# Models
# ============================================================

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int] = [256, 256, 128],
        dropout: float = 0.1,
        use_bn: bool = True,
    ):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(d, 1)

    def forward(self, x):
        feat = self.backbone(x)
        reg = self.out(feat).squeeze(-1)
        return {"feat": feat, "reg": reg}


class TemporalCNN(nn.Module):
    """
    简洁 1D CNN：Conv -> ReLU -> Conv -> ReLU -> GAP -> heads
    输入 (B, L, F) -> 转置为 (B, F, L)
    """
    def __init__(self, in_feats: int, channels: List[int] = [64, 128], ksize: int = 5, dropout: float = 0.1):
        super().__init__()
        c1, c2 = channels
        self.conv1 = nn.Conv1d(in_feats, c1, kernel_size=ksize, padding=ksize//2)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=ksize, padding=ksize//2)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(c1)
        self.bn2 = nn.BatchNorm1d(c2)
        self.out_dim = c2

    def forward(self, x):
        # x: (B, L, F) -> (B, F, L)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)
        
        x = torch.mean(x, dim=-1)
        return x  


class MultiTaskHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.reg = nn.Linear(in_dim, 1)
        self.cls = nn.Linear(in_dim, 1)  

    def forward(self, feat):
        reg = self.reg(feat).squeeze(-1)
        logit = self.cls(feat).squeeze(-1)
        prob = torch.sigmoid(logit)
        return {"reg": reg, "logit": logit, "prob": prob}


class MultiTaskMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float = 0.1, use_bn: bool = True):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h)]
            if use_bn:
                layers += [nn.BatchNorm1d(h)]
            layers += [nn.ReLU()]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        self.backbone = nn.Sequential(*layers)
        self.head = MultiTaskHead(d)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


class MultiTaskTCN(nn.Module):
    def __init__(self, in_feats: int, channels: List[int] = [64, 128], ksize: int = 5, dropout: float = 0.1):
        super().__init__()
        self.backbone = TemporalCNN(in_feats, channels, ksize, dropout)
        self.head = MultiTaskHead(self.backbone.out_dim)

    def forward(self, x_seq):  
        feat = self.backbone(x_seq)
        return self.head(feat)


# ============================================================
# Training (single-output baseline MLP)
# ============================================================

@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 8192
    epochs: int = 20
    hidden_dims: List[int] = None  
    dropout: float = 0.1
    use_bn: bool = True
    loss: str = "huber"  # "huber" | "mse" | "mae"
    huber_delta: float = 1.0
    grad_clip_norm: Optional[float] = 5.0
    early_stop_patience: int = 5
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1223
    log_dir: str = "logs"
    save_dir: str = "checkpoints"


class Trainer:
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: TrainConfig,
        feature_names: Optional[List[str]] = None
    ):
        set_seed(config.seed)
        self.cfg = config
        self.feature_names = feature_names or [f"x{i}" for i in range(X_train.shape[1])]

        
        self.train_ds = NumpyDataset(X_train, y_train)
        self.val_ds = NumpyDataset(X_val, y_val)
        self.train_loader = DataLoader(
            self.train_ds, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, pin_memory=True, drop_last=False
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=True, drop_last=False
        )

        
        hidden = config.hidden_dims if config.hidden_dims is not None else [256, 256, 128]
        self.model = MLP(in_dim=X_train.shape[1], hidden_dims=hidden, dropout=config.dropout, use_bn=config.use_bn)
        self.model.to(config.device)

        
        if config.loss == "huber":
            self.criterion = nn.HuberLoss(delta=config.huber_delta, reduction="mean")
        elif config.loss == "mse":
            self.criterion = nn.MSELoss(reduction="mean")
        elif config.loss == "mae":
            self.criterion = nn.L1Loss(reduction="mean")
        else:
            raise ValueError("Unknown loss")

        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        
        run_name = f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.run_name = run_name
        self.log_dir  = os.path.join(config.log_dir,  run_name)
        self.save_dir = os.path.join(config.save_dir, run_name)
        os.makedirs(self.log_dir,  exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.log_path = os.path.join(self.log_dir, "train_log.csv")
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,train_mae,val_loss,val_mae,val_rmse,val_pearson,val_spearman\n")

        
        with open(os.path.join(self.save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, ensure_ascii=False, indent=2)

        self.best_val  = float("inf")
        self.best_path = os.path.join(self.save_dir, "best_model.pt")
        self.last_path = os.path.join(self.save_dir, "last_model.pt")
        self.no_improve = 0

    def _one_epoch(self, train=True):
        loader = self.train_loader if train else self.val_loader
        device = self.cfg.device
        model = self.model
        criterion = self.criterion

        model.train(mode=train)
        total_loss = 0.0
        total_mae = 0.0
        n = 0

        pbar = tqdm(loader, desc="train" if train else "val", leave=False)
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.set_grad_enabled(train):
                pred_dict = model(xb)
                pred = pred_dict["reg"]
                loss = criterion(pred, yb)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.cfg.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip_norm)
                    self.optimizer.step()

            bs = yb.size(0)
            total_loss += loss.item() * bs
            total_mae += torch.mean(torch.abs(pred.detach() - yb)).item() * bs
            n += bs
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = total_loss / max(n, 1)
        mean_mae = total_mae / max(n, 1)
        return mean_loss, mean_mae

    def fit(self, X_val: np.ndarray, y_val: np.ndarray):
        device = self.cfg.device
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss, train_mae = self._one_epoch(train=True)
            val_loss, val_mae = self._one_epoch(train=False)

            
            self.model.eval()
            preds = []
            with torch.no_grad():
                for xb, yb in self.val_loader:
                    xb = xb.to(device, non_blocking=True)
                    pred = self.model(xb)["reg"].detach().cpu().numpy()
                    preds.append(pred)
            y_pred = np.concatenate(preds, axis=0)
            y_true = y_val

            val_rmse = compute_rmse(y_true, y_pred)
            corrs = compute_corrs(y_true, y_pred)

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch},{train_loss:.6f},{train_mae:.6f},{val_loss:.6f},{val_mae:.6f},{val_rmse:.6f},{corrs['pearson']:.6f},{corrs['spearman']:.6f}\n")

            tqdm.write(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_mae {val_mae:.4f} | val_rmse {val_rmse:.4f} | val_P {corrs['pearson']:.4f} | val_S {corrs['spearman']:.4f}")

            improve = val_loss < self.best_val - 1e-6
            if improve:
                self.best_val = val_loss
                self.no_improve = 0
                torch.save(self.model.state_dict(), self.best_path)
            else:
                self.no_improve += 1

            torch.save(self.model.state_dict(), self.last_path)

            if self.no_improve >= self.cfg.early_stop_patience:
                tqdm.write(f"Early stopping at epoch {epoch}. Best val_loss={self.best_val:.6f}")
                break

        return self.best_path, self.last_path

    def predict(self, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        ds = NumpyDataset(X, None)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=self.cfg.num_workers, pin_memory=True)
        device = self.cfg.device
        self.model.eval()
        preds = []
        with torch.no_grad():
            for xb in tqdm(dl, desc="predict", leave=False):
                xb = xb.to(device, non_blocking=True)
                pred = self.model(xb)["reg"].detach().cpu().numpy()
                preds.append(pred)
        return np.concatenate(preds, axis=0)


# ============================================================
# Trainer for Multi-task + Sequence
# ============================================================

@dataclass
class MTSeqConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4096
    epochs: int = 20
    channels: List[int] = None   
    ksize: int = 5
    dropout: float = 0.1
    grad_clip_norm: Optional[float] = 5.0
    early_stop_patience: int = 5
    num_workers: int = 0
    use_amp: bool = True
    amp_dtype: str = "bf16"      
    grad_accum_steps: int = 1    
    compile_model: bool = True   
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1223
    log_dir: str = "logs"
    save_dir: str = "checkpoints"
    alpha_reg: float = 1.0       # 回归损失权重（非零样本）
    alpha_cls: float = 1.0       # 分类损失权重
    pos_weight: Optional[float] = None  


class MTSeqTrainer:
    def __init__(self, X_train_seq: np.ndarray, y_train: np.ndarray, X_val_seq: np.ndarray, y_val: np.ndarray, cfg: MTSeqConfig):
        set_seed(cfg.seed)
        self.cfg = cfg

        self.train_ds = SeqDataset(X_train_seq, y_train)
        self.val_ds   = SeqDataset(X_val_seq,   y_val)
        self.train_loader = DataLoader(self.train_ds, batch_size=cfg.batch_size, shuffle=True,
                                       num_workers=cfg.num_workers, pin_memory=True)
        self.val_loader   = DataLoader(self.val_ds,   batch_size=cfg.batch_size, shuffle=False,
                                       num_workers=cfg.num_workers, pin_memory=True)

        in_feats = X_train_seq.shape[-1]
        channels = cfg.channels if cfg.channels is not None else [64, 128]
        self.model = MultiTaskTCN(in_feats=in_feats, channels=channels, ksize=cfg.ksize, dropout=cfg.dropout).to(cfg.device)

        
        self.huber = nn.HuberLoss(delta=1.0, reduction="mean")
        if cfg.pos_weight is None:
            
            p_pos = max(1e-6, float(np.mean(y_train != 0)))
            pos_weight = (1.0 - p_pos) / max(p_pos, 1e-6)
        else:
            pos_weight = cfg.pos_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=cfg.device))

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        
        run_name = f"seq_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.run_name = run_name
        self.log_dir  = os.path.join(cfg.log_dir,  run_name)
        self.save_dir = os.path.join(cfg.save_dir, run_name)
        os.makedirs(self.log_dir,  exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, "train_log.csv")
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss,val_rmse,val_P,val_S,auc,prauc,top5_hit,bottom5_hit\n")

        self.best_val = float("inf")
        self.best_path = os.path.join(self.save_dir, "best_model.pt")
        self.last_path = os.path.join(self.save_dir, "last_model.pt")
        self.no_improve = 0
        if self.cfg.compile_model:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")  
            except Exception as e:
                print("torch.compile 跳过：", e)

        torch.backends.cudnn.benchmark = True  
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    def _loss(self, out: dict, yb: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        
        y_cls = (yb != 0).float()
        
        mask = (y_cls > 0.5)
        if mask.any():
            reg_loss = self.huber(out["reg"][mask], yb[mask])
        else:
            reg_loss = torch.tensor(0.0, device=yb.device)
        cls_loss = self.bce(out["logit"], y_cls)
        total = self.cfg.alpha_reg * reg_loss + self.cfg.alpha_cls * cls_loss
        return total, {"reg": reg_loss.item(), "cls": cls_loss.item()}

    def _one_epoch(self, train=True):
        loader = self.train_loader if train else self.val_loader
        device = self.cfg.device
        model = self.model

        model.train(mode=train)
        total_loss, n = 0.0, 0

        amp_dtype = torch.bfloat16 if self.cfg.amp_dtype.lower()=="bf16" else torch.float16
        scaler = None

        pbar = tqdm(loader, desc="train" if train else "val", leave=False)

        if train:
            self.opt.zero_grad(set_to_none=True)

        for step, (xb, yb) in enumerate(pbar, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.set_grad_enabled(train):
                if self.cfg.use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        out = model(xb)
                        loss, parts = self._loss(out, yb)
                        loss = loss / max(1, self.cfg.grad_accum_steps)
                else:
                    out = model(xb)
                    loss, parts = self._loss(out, yb)
                    loss = loss / max(1, self.cfg.grad_accum_steps)

                if train:
                    if self.cfg.use_amp and self.cfg.amp_dtype.lower()=="fp16":
                        if scaler is None:
                            scaler = torch.cuda.amp.GradScaler()
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if (step % max(1, self.cfg.grad_accum_steps)) == 0:
                        if self.cfg.grad_clip_norm is not None:
                            if scaler is not None:
                                scaler.unscale_(self.opt)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip_norm)

                        if scaler is not None:
                            scaler.step(self.opt); scaler.update()
                        else:
                            self.opt.step()
                        self.opt.zero_grad(set_to_none=True)

            bs = yb.size(0)
            total_loss += loss.item() * bs * max(1, self.cfg.grad_accum_steps)
            n += bs
            pbar.set_postfix(loss=f"{(loss.item()*max(1,self.cfg.grad_accum_steps)):.4f}")

        return total_loss / max(n, 1)

    def fit(self, X_val_seq: np.ndarray, y_val: np.ndarray):
        device = self.cfg.device
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._one_epoch(train=True)
            val_loss   = self._one_epoch(train=False)

            
            self.model.eval()
            preds_reg, preds_prob = [], []
            with torch.no_grad():
                for xb, yb in self.val_loader:
                    xb = xb.to(device, non_blocking=True)
                    out = self.model(xb)
                    preds_reg.append(out["reg"].detach().cpu().numpy())
                    preds_prob.append(out["prob"].detach().cpu().numpy())
            reg = np.concatenate(preds_reg, 0)
            prob = np.concatenate(preds_prob, 0)
            y_true = y_val
            y_hat = prob * reg  

            rmse = compute_rmse(y_true, y_hat)
            corrs = compute_corrs(y_true, y_hat)
            top5, bottom5 = topk_hit_rate(y_true, y_hat, k=0.05)

            auc = prauc = np.nan
            if SKLEARN_AVAILABLE:
                y_bin = (y_true != 0).astype(np.float32)
                try:
                    auc = float(roc_auc_score(y_bin, prob))
                except Exception:
                    pass
                try:
                    prauc = float(average_precision_score(y_bin, prob))
                except Exception:
                    pass

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{rmse:.6f},{corrs['pearson']:.6f},{corrs['spearman']:.6f},{auc:.6f},{prauc:.6f},{top5:.6f},{bottom5:.6f}\n")

            tqdm.write(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                       f"val_rmse {rmse:.4f} | val_P {corrs['pearson']:.4f} | val_S {corrs['spearman']:.4f} | "
                       f"AUC {auc:.4f} | PR-AUC {prauc:.4f} | top5 {top5:.3f} | bottom5 {bottom5:.3f}")

            improve = val_loss < self.best_val - 1e-6
            if improve:
                self.best_val = val_loss
                self.no_improve = 0
                torch.save(self.model.state_dict(), self.best_path)
            else:
                self.no_improve += 1

            torch.save(self.model.state_dict(), self.last_path)
            if self.no_improve >= self.cfg.early_stop_patience:
                tqdm.write(f"Early stopping at epoch {epoch}. Best val_loss={self.best_val:.6f}")
                break

        return self.best_path, self.last_path

    def predict(self, X_seq: np.ndarray, batch_size: int = 4096) -> Dict[str, np.ndarray]:
        ds = SeqDataset(X_seq, None)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=self.cfg.num_workers, pin_memory=True)
        device = self.cfg.device
        self.model.eval()
        regs, probs = [], []
        with torch.no_grad():
            for xb in tqdm(dl, desc="predict", leave=False):
                xb = xb.to(device, non_blocking=True)
                out = self.model(xb)
                regs.append(out["reg"].detach().cpu().numpy())
                probs.append(out["prob"].detach().cpu().numpy())
        reg = np.concatenate(regs, 0)
        prob = np.concatenate(probs, 0)
        return {"reg": reg, "prob": prob, "yhat": prob * reg}


# ============================================================
# High-level helpers
# ============================================================

def train_mlp_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    cfg: Optional[TrainConfig] = None,
    feature_names: Optional[List[str]] = None,
    save_preds: bool = True,
    out_dir: str = "artifacts",
    return_trainer: bool=False
) -> Dict[str, str] | Tuple[Dict[str,str], "Trainer"]:
    """
    单头 MLP 训练：回归 y。
    """
    os.makedirs(out_dir, exist_ok=True)

    if cfg is None:
        cfg = TrainConfig()
    cfg.log_dir = os.path.join(out_dir, "logs")
    cfg.save_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.save_dir, exist_ok=True)

    trainer = Trainer(X_train, y_train, X_val, y_val, cfg, feature_names=feature_names)
    trainer.fit(X_val, y_val)

    paths = {
        "run_name": trainer.run_name,
        "log_dir":  trainer.log_dir,
        "save_dir": trainer.save_dir,
        "log_csv":  trainer.log_path,
        "best_model": trainer.best_path,
        "last_model": trainer.last_path,
    }
    pred_dir = os.path.join(out_dir, trainer.run_name)
    os.makedirs(pred_dir, exist_ok=True)
    if save_preds:
        y_pred_val  = trainer.predict(X_val, batch_size=cfg.batch_size)
        y_pred_test = trainer.predict(X_test, batch_size=cfg.batch_size)
        np.save(os.path.join(pred_dir, "pred_val.npy"), y_pred_val)
        np.save(os.path.join(pred_dir, "pred_test.npy"), y_pred_test)
        np.savetxt(os.path.join(pred_dir, "pred_val.csv"),  y_pred_val,  delimiter=",", fmt="%.6f")
        np.savetxt(os.path.join(pred_dir, "pred_test.csv"), y_pred_test, delimiter=",", fmt="%.6f")
        paths["pred_val"]  = os.path.join(pred_dir, "pred_val.csv")
        paths["pred_test"] = os.path.join(pred_dir, "pred_test.csv")

    meta = {
        "in_dim": int(X_train.shape[1]),
        "train_size": int(X_train.shape[0]),
        "val_size": int(X_val.shape[0]),
        "test_size": int(X_test.shape[0]),
        "config": asdict(cfg),
        "feature_names": feature_names or [f"x{i}" for i in range(X_train.shape[1])]
    }
    with open(os.path.join(pred_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    paths["run_meta"] = os.path.join(pred_dir, "run_meta.json")

    if return_trainer:
        return paths, trainer
    return paths


def train_seq_multitask(
    X_train_seq: np.ndarray, y_train: np.ndarray,
    X_val_seq: np.ndarray,   y_val: np.ndarray,
    X_test_seq: np.ndarray,
    cfg: Optional[MTSeqConfig] = None,
    out_dir: str = "artifacts_seq",
    save_preds: bool = True,
):
    """
    多头（分类 + 回归）+ CNN 时序模型
    - 分类：预测非零概率
    - 回归：仅在非零上拟合幅度（Huber）
    - 合成预测：yhat = prob * reg
    """
    os.makedirs(out_dir, exist_ok=True)
    if cfg is None:
        cfg = MTSeqConfig()

    cfg.log_dir = os.path.join(out_dir, "logs")
    cfg.save_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.save_dir, exist_ok=True)

    trainer = MTSeqTrainer(X_train_seq, y_train, X_val_seq, y_val, cfg)
    trainer.fit(X_val_seq, y_val)

    paths = {
        "run_name": trainer.run_name,
        "log_dir":  trainer.log_dir,
        "save_dir": trainer.save_dir,
        "log_csv":  trainer.log_path,
        "best_model": trainer.best_path,
        "last_model": trainer.last_path,
    }

    pred_dir = os.path.join(out_dir, trainer.run_name)
    os.makedirs(pred_dir, exist_ok=True)
    if save_preds:
        val_pred  = trainer.predict(X_val_seq,  batch_size=cfg.batch_size)
        test_pred = trainer.predict(X_test_seq, batch_size=cfg.batch_size)
        for split, pred in [("val", val_pred), ("test", test_pred)]:
            np.save(os.path.join(pred_dir, f"{split}_reg.npy"),  pred["reg"])
            np.save(os.path.join(pred_dir, f"{split}_prob.npy"), pred["prob"])
            np.save(os.path.join(pred_dir, f"{split}_yhat.npy"), pred["yhat"])
            np.savetxt(os.path.join(pred_dir, f"{split}_yhat.csv"), pred["yhat"], delimiter=",", fmt="%.6f")
        paths["pred_val_yhat"]  = os.path.join(pred_dir, "val_yhat.csv")
        paths["pred_test_yhat"] = os.path.join(pred_dir, "test_yhat.csv")

    return paths

class DPORankWrapper(nn.Module):
    def __init__(self, model: nn.Module, score_fn):
        super().__init__()
        self.policy = model
        self.score_fn = score_fn
        self.ref_policy = copy.deepcopy(model).eval()
        for p in self.ref_policy.parameters():
            p.requires_grad = False

    def forward_score(self, xb):
        return self.score_fn(self.policy, xb)

    @torch.no_grad()
    def forward_score_ref(self, xb):
        return self.score_fn(self.ref_policy, xb)


def dpo_rank_loss(s_w, s_l, s0_w, s0_l, beta=0.1, reg_lambda=0.01):
    """
    s_w, s_l: policy 的 winner/loser 分数 (B,)
    s0_w, s0_l: ref policy 的 winner/loser 分数 (B,)
    """
    d_theta = s_w - s_l
    d_ref   = s0_w - s0_l
    z = beta * (d_theta - d_ref)
    loss_pair = torch.nn.functional.softplus(-z).mean()
    reg = reg_lambda * ( (s_w - s0_w).pow(2).mean() + (s_l - s0_l).pow(2).mean() )
    return loss_pair + reg


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, X, winner_idx, loser_idx, device="cpu"):
        assert len(winner_idx) == len(loser_idx)
        self.X = X
        self.w = winner_idx.astype(np.int64)
        self.l = loser_idx.astype(np.int64)
        self.device = device

    def __len__(self):
        return len(self.w)

    def __getitem__(self, i):
        iw, il = self.w[i], self.l[i]
        xw = torch.as_tensor(self.X[iw], dtype=torch.float32)
        xl = torch.as_tensor(self.X[il], dtype=torch.float32)
        return xw, xl


def train_dpo_rank(
    base_model: nn.Module,
    score_fn,           
    X_feat, y_true,     
    day_vec=None,       
    session_vec=None,   
    margin=0.5,         
    max_pairs_per_day=20000,
    epochs=2,
    beta=0.1,
    reg_lambda=0.01,
    lr=1e-4,
    batch_size=8192,
    use_amp=True,
    amp_dtype="bf16",
    device="cuda",
    num_workers=0,
    shuffle_pairs=True,
):
    model = DPORankWrapper(base_model, score_fn).to(device)
    opt = optim.AdamW(model.policy.parameters(), lr=lr, weight_decay=1e-4)

    y = y_true.copy()
    idx_all = np.arange(len(y))

    
    if day_vec is not None:
        key = day_vec.astype(str)
        if session_vec is not None:
            key = np.char.add(key, "_"+session_vec.astype(str))
    else:
        key = np.array(["all"]*len(y))

    winners, losers = [], []
    rng = np.random.RandomState(42)

    for g in np.unique(key):
        mask = (key == g)
        idx_g = idx_all[mask]
        if len(idx_g) < 2: 
            continue
        y_g = y[mask]
        pos = idx_g[y_g >  0 + 1e-12]
        nonpos = idx_g[y_g <= 0 + 1e-12]
        if len(pos) > 0 and len(nonpos) > 0:
            k = min(max_pairs_per_day//2, len(pos)*4)  
            wi = rng.choice(pos,    size=min(k, len(pos)),    replace=True)
            li = rng.choice(nonpos, size=min(k, len(nonpos)), replace=True)
            winners.append(wi); losers.append(li)

        
        big = idx_g[np.abs(y_g) >= (margin)]
        small = idx_g[(np.abs(y_g) <  (margin))]
        if len(big) > 0 and len(small) > 0:
            k = min(max_pairs_per_day//2, len(big)*4)
            wi = rng.choice(big,   size=min(k, len(big)),   replace=True)
            li = rng.choice(small, size=min(k, len(small)), replace=True)
            winners.append(wi); losers.append(li)

    if len(winners) == 0:
        raise RuntimeError("len(winners) == 0")

    winner_idx = np.concatenate(winners)
    loser_idx  = np.concatenate(losers)
    if shuffle_pairs:
        perm = rng.permutation(len(winner_idx))
        winner_idx = winner_idx[perm]; loser_idx = loser_idx[perm]

    ds = PairDataset(X_feat, winner_idx, loser_idx)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    
    amp_dtype_t = torch.bfloat16 if amp_dtype.lower() == "bf16" else torch.float16
    model.train()
    for ep in range(1, epochs+1):
        tot, n = 0.0, 0
        pbar = tqdm(dl, desc=f"DPO-rank ep{ep}", leave=False)
        for xw, xl in pbar:
            xw = xw.to(device, non_blocking=True)
            xl = xl.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype_t):
                    s_w  = model.forward_score(xw).view(-1)
                    s_l  = model.forward_score(xl).view(-1)
                    s0_w = model.forward_score_ref(xw).view(-1)
                    s0_l = model.forward_score_ref(xl).view(-1)
                    loss = dpo_rank_loss(s_w, s_l, s0_w, s0_l, beta=beta, reg_lambda=reg_lambda)
            else:
                s_w  = model.forward_score(xw).view(-1)
                s_l  = model.forward_score(xl).view(-1)
                s0_w = model.forward_score_ref(xw).view(-1)
                s0_l = model.forward_score_ref(xl).view(-1)
                loss = dpo_rank_loss(s_w, s_l, s0_w, s0_l, beta=beta, reg_lambda=reg_lambda)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 5.0)
            opt.step()

            bs = xw.size(0)
            tot += loss.item() * bs
            n   += bs
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"[DPO-rank] epoch {ep} loss={tot/max(1,n):.6f}")

    return model.policy