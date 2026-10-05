"""Selected unchanged methods from the later Amber research archive.

These helpers are not a complete reproduction of the research study.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 5:
        return np.nan
    x0 = x[mask] - x[mask].mean()
    y0 = y[mask] - y[mask].mean()
    denominator = np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0))
    if denominator <= 1e-12:
        return np.nan
    return float(np.sum(x0 * y0) / denominator)

def bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 10:
        return np.nan, np.nan
    values = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(x), len(x))
        value = pearson_r(x[indices], y[indices])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return np.nan, np.nan
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))

class RidgeRegressor:
    """Median-imputed, standardized Ridge with the original alpha of 1.0."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "RidgeRegressor":
        values = features.to_numpy(dtype=float)
        values[~np.isfinite(values)] = np.nan
        self.medians = np.nanmedian(values, axis=0)
        self.medians[~np.isfinite(self.medians)] = 0.0
        values = np.where(np.isnan(values), self.medians, values)
        self.means = values.mean(axis=0)
        self.scales = values.std(axis=0, ddof=0)
        self.scales[self.scales <= 1e-12] = 1.0
        standardized = (values - self.means) / self.scales
        y = target.to_numpy(dtype=float)
        self.intercept = float(y.mean())
        centered_y = y - self.intercept
        gram = standardized.T @ standardized
        penalty = self.alpha * np.eye(standardized.shape[1])
        self.coef = np.linalg.solve(gram + penalty, standardized.T @ centered_y)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        values = features.to_numpy(dtype=float)
        values[~np.isfinite(values)] = np.nan
        values = np.where(np.isnan(values), self.medians, values)
        standardized = (values - self.means) / self.scales
        return self.intercept + standardized @ self.coef

def assign_positions(samples: pd.DataFrame) -> pd.DataFrame:
    work = samples.sort_values(["institute_year", "user_id", "target_name", "date"]).copy()
    groups = work.groupby(["institute_year", "user_id", "target_name"], sort=False)
    work["trajectory_rank"] = groups.cumcount().astype(int)
    work["trajectory_n"] = groups["date"].transform("size").astype(int)
    return work

def block_mask(samples: pd.DataFrame, regime: str, block: str) -> pd.Series:
    rank = samples["trajectory_rank"]
    count = samples["trajectory_n"]
    if regime == "expanding_refit":
        train_end = np.floor(0.65 * count).astype(int)
        early_end = train_end + np.ceil((count - train_end) / 2).astype(int)
        bounds = {
            "early_train": (np.zeros(len(samples), dtype=int), train_end),
            "early_test": (train_end, early_end),
            "late_train": (np.zeros(len(samples), dtype=int), early_end),
            "late_test": (early_end, count),
        }
    elif regime == "disjoint_refit":
        cut_35 = np.floor(0.35 * count).astype(int)
        cut_50 = np.floor(0.50 * count).astype(int)
        cut_85 = np.floor(0.85 * count).astype(int)
        bounds = {
            "early_train": (np.zeros(len(samples), dtype=int), cut_35),
            "early_test": (cut_35, cut_50),
            "late_train": (cut_50, cut_85),
            "late_test": (cut_85, count),
        }
    else:
        raise ValueError(regime)
    low, high = bounds[block]
    return rank.ge(low) & rank.lt(high)
