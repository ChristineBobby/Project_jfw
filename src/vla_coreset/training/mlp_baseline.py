from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def make_feature_matrix(image_features: np.ndarray, text_features: np.ndarray) -> np.ndarray:
    image_features = np.asarray(image_features, dtype=np.float32)
    text_features = np.asarray(text_features, dtype=np.float32)
    if image_features.ndim != 2:
        raise ValueError(f"Expected image features [N,D], got {image_features.shape}")
    if text_features.shape != (1, text_features.shape[-1]):
        raise ValueError(f"Expected one text feature row, got {text_features.shape}")
    repeated_text = np.repeat(text_features, image_features.shape[0], axis=0)
    return np.concatenate([image_features, repeated_text], axis=1).astype(np.float32)


@dataclass(frozen=True)
class ActionNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, actions: np.ndarray) -> "ActionNormalizer":
        actions = np.asarray(actions, dtype=np.float32)
        mean = actions.mean(axis=0)
        std = actions.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, actions: np.ndarray) -> np.ndarray:
        return ((np.asarray(actions, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, actions: np.ndarray) -> np.ndarray:
        return (np.asarray(actions, dtype=np.float32) * self.std + self.mean).astype(np.float32)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 7, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_mse_metrics(pred: np.ndarray, target: np.ndarray, prefix: str) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {target.shape}")
    per_dim = ((pred - target) ** 2).mean(axis=0)
    metrics = {f"{prefix}_mse": float(per_dim.mean())}
    metrics.update({f"{prefix}_action_{i}_mse": float(value) for i, value in enumerate(per_dim)})
    return metrics


def load_feature_artifacts(feature_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    features = np.load(feature_dir / "features.npy")
    actions = np.load(feature_dir / "actions_left7.npy")
    text_features = np.load(feature_dir / "text_features.npy")
    index = pd.read_parquet(feature_dir / "index.parquet")
    return features, actions, text_features, index


def mask_for_episodes(index: pd.DataFrame, episodes: list[int]) -> np.ndarray:
    return index["episode_index"].isin(episodes).to_numpy()


def mask_for_split(index: pd.DataFrame, split_name: str) -> np.ndarray:
    return (index["split"] == split_name).to_numpy()


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def _predict(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    loader = _make_loader(x, np.zeros((len(x), 7), dtype=np.float32), batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        for batch_x, _ in loader:
            pred = model(batch_x.to(device, non_blocking=True))
            preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def train_mlp_baseline(
    x: np.ndarray,
    actions: np.ndarray,
    index: pd.DataFrame,
    selected_episodes: list[int],
    seed: int,
    device_name: str,
    batch_size: int,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    progress: bool = True,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_mask = mask_for_episodes(index, selected_episodes)
    val_mask = mask_for_split(index, "val")
    test_mask = mask_for_split(index, "test")
    normalizer = ActionNormalizer.fit(actions[train_mask])
    y_train = normalizer.transform(actions[train_mask])
    y_val = normalizer.transform(actions[val_mask])
    y_test = normalizer.transform(actions[test_mask])

    x_train = x[train_mask].astype(np.float32)
    x_val = x[val_mask].astype(np.float32)
    x_test = x[test_mask].astype(np.float32)

    model = MLPRegressor(input_dim=x.shape[1], output_dim=actions.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    train_loader = _make_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(x_val, y_val, batch_size=batch_size, shuffle=False)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    iterator = range(1, max_epochs + 1)
    if progress:
        iterator = tqdm(iterator, desc=f"seed {seed}", leave=False)

    for epoch in iterator:
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x.to(device, non_blocking=True))
            loss = loss_fn(pred, batch_y.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()

        val_losses = []
        model.eval()
        with torch.inference_mode():
            for batch_x, batch_y in val_loader:
                pred = model(batch_x.to(device, non_blocking=True))
                val_losses.append(loss_fn(pred, batch_y.to(device, non_blocking=True)).item())
        val_loss = float(np.mean(val_losses))
        if progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(val_mse=f"{val_loss:.5f}", best=f"{best_val:.5f}")
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    val_pred_norm = _predict(model, x_val, batch_size=batch_size, device=device)
    test_pred_norm = _predict(model, x_test, batch_size=batch_size, device=device)
    val_pred = normalizer.inverse_transform(val_pred_norm)
    test_pred = normalizer.inverse_transform(test_pred_norm)
    y_val_orig = actions[val_mask]
    y_test_orig = actions[test_mask]

    metrics = {
        "seed": int(seed),
        "selected_episodes": " ".join(str(ep) for ep in selected_episodes),
        "train_frames": int(train_mask.sum()),
        "val_frames": int(val_mask.sum()),
        "test_frames": int(test_mask.sum()),
        "best_epoch": int(best_epoch),
        "best_val_normalized_mse": float(best_val),
    }
    metrics.update(compute_mse_metrics(val_pred_norm, y_val, prefix="val_normalized"))
    metrics.update(compute_mse_metrics(test_pred_norm, y_test, prefix="test_normalized"))
    metrics.update(compute_mse_metrics(val_pred, y_val_orig, prefix="val_original"))
    metrics.update(compute_mse_metrics(test_pred, y_test_orig, prefix="test_original"))
    return {
        "metrics": metrics,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "val_target": y_val_orig,
        "test_target": y_test_orig,
    }
