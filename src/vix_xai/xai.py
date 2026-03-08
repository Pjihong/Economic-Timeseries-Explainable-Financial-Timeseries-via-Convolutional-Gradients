"""
xai.py — Explainability: Grad-CAM, embedding extraction, CPD evaluation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader


# ═══════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════


def _find_last_conv_per_branch(model: nn.Module) -> Dict[str, nn.Module]:
    """Find the last Conv1d layer in each TCN / CNN branch."""
    out = {}
    for i, tcn in enumerate(model.tcns):
        last_name, last_mod = None, None
        for name, m in tcn.named_modules():
            if isinstance(m, nn.Conv1d):
                last_name, last_mod = name, m
        if last_mod is not None:
            out[f"tcns.{i}.{last_name}"] = last_mod
    if not out:
        raise RuntimeError("Conv1d not found in model branches")
    return out


# ═══════════════════════════════════════════════════════════════════
# Grad-CAM
# ═══════════════════════════════════════════════════════════════════


class TimeSeriesGradCAMRegression:
    """
    Grad-CAM for regression models with per-branch TCN architecture.

    Hooks into the last Conv1d of each branch to compute a (T,) importance map.
    """

    def __init__(self, model: nn.Module, device: torch.device, aggregate: str = "mean"):
        self.model = model
        self.device = device
        self.aggregate = aggregate
        self.target_map = _find_last_conv_per_branch(model)
        self.fmap: Dict[str, torch.Tensor] = {}
        self.grad: Dict[str, torch.Tensor] = {}
        self._handles: list = []
        self._register()

    def _register(self):
        def f_hook(key):
            def _fh(m, inp, out):
                o = out[0] if isinstance(out, (tuple, list)) else out
                self.fmap[key] = o.detach()
            return _fh

        def b_hook(key):
            def _bh(m, gin, gout):
                g = gout[0] if isinstance(gout, (tuple, list)) else gout
                self.grad[key] = g.detach()
            return _bh

        for name, layer in self.target_map.items():
            self._handles.append(layer.register_forward_hook(f_hook(name)))
            self._handles.append(layer.register_full_backward_hook(b_hook(name)))

    def remove_hooks(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    @staticmethod
    def _cam_1layer(fmap: torch.Tensor, grad: torch.Tensor, T: int) -> np.ndarray:
        A, G = fmap[0], grad[0]
        w = G.mean(dim=-1)
        cam = (w.unsqueeze(-1) * A).sum(dim=0)
        cam = F.relu(cam).detach().cpu().numpy()
        if cam.size != T:
            cam = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, cam.size), cam)
        m = cam.max()
        return cam / m if m > 0 else cam

    def generate(
        self,
        x: torch.Tensor,
        smooth_steps: int = 8,
        noise_sigma: float = 0.10,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute aggregated Grad-CAM map for a single sample.

        Parameters
        ----------
        x : (1, T, F) tensor
        smooth_steps : int
            SmoothGrad averaging steps (0 = no smoothing).
        noise_sigma : float
            Noise std for SmoothGrad.

        Returns
        -------
        cam_agg : (T,) aggregated importance
        per_branch : dict[branch_name → (T,) importance]
        """
        assert x.dim() == 3 and x.size(0) == 1
        x = x.to(self.device)
        T = x.size(1)
        was_training = self.model.training
        self.model.eval()

        def one_pass(inp):
            self.model.zero_grad(set_to_none=True)
            y = self.model(inp).squeeze()
            y.backward()
            cams = {}
            for name in self.target_map:
                if name in self.fmap and name in self.grad:
                    cams[name] = self._cam_1layer(self.fmap[name], self.grad[name], T)
            return cams

        try:
            if smooth_steps > 0:
                acc: Dict[str, list] = {}
                for _ in range(smooth_steps):
                    noise = torch.randn_like(x) * noise_sigma
                    cams = one_pass((x + noise).clone().requires_grad_(True))
                    for k, v in cams.items():
                        acc.setdefault(k, []).append(v)
                cams = {k: np.mean(vs, axis=0) for k, vs in acc.items()}
            else:
                cams = one_pass(x.clone().requires_grad_(True))

            stack = np.stack(list(cams.values()), axis=0)
            cam_agg = stack.mean(axis=0) if self.aggregate == "mean" else stack.sum(axis=0)
            cam_agg = cam_agg / cam_agg.max() if cam_agg.max() > 0 else cam_agg
            return cam_agg.astype(np.float32), cams
        finally:
            self.model.train(was_training)


# ═══════════════════════════════════════════════════════════════════
# Data collection helpers
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def collect_test_windows(dl: DataLoader) -> torch.Tensor:
    """Collect all X batches from a DataLoader into a single tensor."""
    return torch.cat([x for x, _ in dl], dim=0)


def inverse_all_X_windows(X_scaled: torch.Tensor, scaler_X) -> np.ndarray:
    """Inverse-transform all windows back to the original scale."""
    N, T, F_ = X_scaled.shape
    flat = X_scaled.numpy().reshape(N * T, F_)
    return scaler_X.inverse_transform(flat).reshape(N, T, F_)


# ═══════════════════════════════════════════════════════════════════
# Late Concatenation: multivariate embedding extraction
# ═══════════════════════════════════════════════════════════════════


def extract_multivariate_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    feature_names: List[str],
    concept_features: List[str],
    device: torch.device | str = "cuda",
) -> np.ndarray:
    """
    Late Concatenation: concatenate per-branch embeddings for selected features.

    Works for both TCNEnsemble and CNNEnsemble.

    Returns
    -------
    combined : (N, C * len(concept_features)) array
    """
    model.eval()
    branch_indices = [
        feature_names.index(feat) for feat in concept_features if feat in feature_names
    ]
    if not branch_indices:
        raise ValueError(f"None of {concept_features} found in feature_names")

    all_combined = []
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            x_normed = model.revin(batch_x, "norm")
            embs = []
            for b_idx in branch_indices:
                emb = model.extract_branch_embedding(x_normed, b_idx)
                embs.append(emb)
            combined = torch.cat(embs, dim=1)
            all_combined.append(combined.cpu().numpy())

    return np.concatenate(all_combined, axis=0)


# ═══════════════════════════════════════════════════════════════════
# CPD (Change Point Detection) evaluation
# ═══════════════════════════════════════════════════════════════════


def evaluate_cpd_performance(
    dates: np.ndarray,
    distance_scores: np.ndarray,
    exogenous_events: List[str],
    window_size: int = 5,
) -> dict:
    """
    Evaluate how well a distance score detects known exogenous shocks.

    Returns dict with AUC, Best_F1, Precision, Recall, Avg_Delay_Days.
    """
    df = pd.DataFrame({"date": pd.to_datetime(dates), "score": distance_scores})
    df["label"] = 0

    for event_date in exogenous_events:
        event_dt = pd.to_datetime(event_date)
        mask = (df["date"] >= event_dt - pd.Timedelta(days=window_size)) & (
            df["date"] <= event_dt + pd.Timedelta(days=window_size)
        )
        df.loc[mask, "label"] = 1

    y_true = df["label"].values
    y_score = df["score"].values

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {
            "AUC": float("nan"),
            "Best_F1": float("nan"),
            "Precision": float("nan"),
            "Recall": float("nan"),
            "Avg_Delay_Days": float("inf"),
        }

    auc = roc_auc_score(y_true, y_score)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1_scores))

    delays: List[float] = []
    best_thr = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    for event_date in exogenous_events:
        event_dt = pd.to_datetime(event_date)
        post = df[
            (df["date"] >= event_dt)
            & (df["date"] <= event_dt + pd.Timedelta(days=window_size))
        ]
        detected = post[post["score"] >= best_thr]
        if not detected.empty:
            delays.append((detected["date"].iloc[0] - event_dt).days)

    return {
        "AUC": float(auc),
        "Best_F1": float(f1_scores[best_idx]),
        "Precision": float(precisions[best_idx]),
        "Recall": float(recalls[best_idx]),
        "Avg_Delay_Days": float(np.mean(delays)) if delays else float("inf"),
    }
