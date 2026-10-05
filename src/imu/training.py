"""Training utilities for Stage 1."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WindowTensorDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[index], self.labels[index]


@dataclass(slots=True)
class EpochResult:
    loss: float
    macro_f1: float
    report: dict[str, Any]


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def create_sampler(labels: np.ndarray, sample_weight_multipliers: np.ndarray | None = None) -> WeightedRandomSampler:
    class_counts = np.bincount(labels)
    sample_weights = 1.0 / class_counts[labels]
    if sample_weight_multipliers is not None:
        sample_weights = sample_weights * sample_weight_multipliers
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def make_loader(
    windows: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
    balanced: bool = False,
    sample_weight_multipliers: np.ndarray | None = None,
) -> DataLoader:
    dataset = WindowTensorDataset(windows, labels)
    sampler = create_sampler(labels, sample_weight_multipliers=sample_weight_multipliers) if balanced else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
) -> EpochResult:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        losses.append(float(loss.detach().cpu()))
        predictions.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    report = classification_report(targets, predictions, output_dict=True, zero_division=0)
    return EpochResult(loss=float(np.mean(losses)), macro_f1=float(macro_f1), report=report)


def fit_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    learning_rate: float,
    epochs: int,
    label_smoothing: float,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=label_smoothing)
    history: list[dict[str, Any]] = []
    best_state = None
    best_f1 = -1.0

    model.to(device)
    for epoch in range(1, epochs + 1):
        train_result = run_epoch(model, train_loader, optimizer, criterion, device)
        with torch.no_grad():
            val_result = run_epoch(model, val_loader, None, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result.loss,
                "train_macro_f1": train_result.macro_f1,
                "val_loss": val_result.loss,
                "val_macro_f1": val_result.macro_f1,
            }
        )
        if val_result.macro_f1 > best_f1:
            best_f1 = val_result.macro_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint")
    model.load_state_dict(best_state)
    return model, history
