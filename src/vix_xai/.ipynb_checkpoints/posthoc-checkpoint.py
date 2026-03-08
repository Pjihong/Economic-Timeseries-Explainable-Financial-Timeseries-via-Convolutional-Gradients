"""
posthoc.py — Enhanced post-hoc analysis (event/control CAM, matching, deletion, permutation).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats as sp_stats
    from scipy.ndimage import label as nd_label
except ImportError:
    sp_stats = None
    nd_label = None


# ═══════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    if n == 0:
        return pvals.copy()
    order = np.argsort(pvals)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)
    corrected = pvals * n / ranked
    corrected = np.minimum.accumulate(corrected[np.argsort(-ranked)])
    corrected = corrected[np.argsort(np.argsort(ranked).astype(int))]
    return np.clip(corrected, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════
# 1) Event definition
# ═══════════════════════════════════════════════════════════════════


def define_events_from_level(
    level: np.ndarray,
    index: pd.Index,
    horizon: int = 5,
    q_event: float = 0.95,
    q_control_range: Tuple[float, float] = (0.4, 0.6),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (future_max_abs_change, event_mask, control_mask)."""
    level = np.asarray(level, dtype=np.float64).ravel()
    n = len(level)
    max_changes = np.full(n, np.nan)
    for i in range(n - horizon):
        fut = level[i + 1 : i + 1 + horizon]
        max_changes[i] = np.abs(fut - level[i]).max()
    valid = max_changes[np.isfinite(max_changes)]
    if len(valid) == 0:
        return max_changes, np.zeros(n, bool), np.zeros(n, bool)
    thr_ev = np.quantile(valid, q_event)
    thr_lo = np.quantile(valid, q_control_range[0])
    thr_hi = np.quantile(valid, q_control_range[1])
    return max_changes, max_changes >= thr_ev, (max_changes >= thr_lo) & (max_changes <= thr_hi)


def build_event_definition_df(
    level: np.ndarray,
    dates: np.ndarray,
    horizon: int,
    q_event: float,
    q_control_range: Tuple[float, float] = (0.4, 0.6),
) -> pd.DataFrame:
    max_ch, ev, ct = define_events_from_level(level, pd.Index(dates), horizon, q_event, q_control_range)
    return pd.DataFrame({
        "date": dates, "event_label": ev.astype(int), "control_label": ct.astype(int),
        "future_max_abs_change": max_ch, "horizon": horizon, "q_event": q_event,
    })


# ═══════════════════════════════════════════════════════════════════
# 2) Non-overlap subsampling
# ═══════════════════════════════════════════════════════════════════


def subsample_nonoverlap(indices: np.ndarray, min_gap: int) -> np.ndarray:
    if len(indices) == 0:
        return indices
    indices = np.sort(indices)
    kept = [indices[0]]
    for idx in indices[1:]:
        if idx - kept[-1] >= min_gap:
            kept.append(idx)
    return np.array(kept, dtype=int)


# ═══════════════════════════════════════════════════════════════════
# 3) Matching + Balance (SMD)
# ═══════════════════════════════════════════════════════════════════


def _compute_smd(a: np.ndarray, b: np.ndarray) -> float:
    ma, mb = a.mean(), b.mean()
    sa, sb = a.std() + 1e-12, b.std() + 1e-12
    return float((ma - mb) / np.sqrt((sa ** 2 + sb ** 2) / 2))


def match_and_report(
    event_idx: np.ndarray,
    ctrl_idx: np.ndarray,
    feats: np.ndarray,
    method: str = "knn",
    k_neighbors: int = 1,
    feature_names_for_balance: Optional[List[str]] = None,
) -> Tuple[list, pd.DataFrame]:
    scaler = StandardScaler()
    Z = scaler.fit_transform(feats)
    Ze, Zc = Z[event_idx], Z[ctrl_idx]

    if method == "propensity":
        X_all = np.vstack([Ze, Zc])
        y_all = np.array([1] * len(Ze) + [0] * len(Zc))
        clf = LogisticRegression(solver="liblinear", max_iter=500).fit(X_all, y_all)
        ps = clf.predict_proba(X_all)[:, 1]
        pe, pc = ps[:len(Ze)].reshape(-1, 1), ps[len(Ze):].reshape(-1, 1)
        nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(pc)
        dists, idxs = nbrs.kneighbors(pe)
    else:
        nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(Zc)
        dists, idxs = nbrs.kneighbors(Ze)

    pairs = []
    for i, (nn_idx, nn_dist) in enumerate(zip(idxs, dists)):
        pairs.append((int(event_idx[i]), [int(ctrl_idx[j]) for j in nn_idx], float(nn_dist.mean())))

    matched_ctrl = np.array([p[1][0] for p in pairs], dtype=int)
    Fe_before, Fc_before = feats[event_idx], feats[ctrl_idx]
    Fe_after = feats[np.array([p[0] for p in pairs])]
    Fc_after = feats[matched_ctrl]

    n_feat = feats.shape[1]
    names = feature_names_for_balance or [f"feat_{i}" for i in range(n_feat)]
    while len(names) < n_feat:
        names.append(f"feat_{len(names)}")

    rows = []
    for j in range(n_feat):
        rows.append({
            "feature_name": names[j] if j < len(names) else f"feat_{j}",
            "mean_event_before": float(Fe_before[:, j].mean()),
            "mean_control_before": float(Fc_before[:, j].mean()),
            "smd_before": _compute_smd(Fe_before[:, j], Fc_before[:, j]),
            "mean_event_after": float(Fe_after[:, j].mean()),
            "mean_control_after": float(Fc_after[:, j].mean()),
            "smd_after": _compute_smd(Fe_after[:, j], Fc_after[:, j]),
            "match_method": method,
        })
    return pairs, pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# 4) CAM helpers
# ═══════════════════════════════════════════════════════════════════


def _cam_stats(cam: np.ndarray, last_k: int = 5) -> dict:
    cam = np.asarray(cam, dtype=np.float64).ravel()
    T = len(cam)
    t = np.arange(T, dtype=np.float64)
    cam_abs = np.abs(cam)
    a_sum = cam_abs.sum() + 1e-12
    return dict(
        com_signed=float((t * cam).sum() / (np.abs(cam).sum() + 1e-12)),
        com_abs=float((t * cam_abs).sum() / a_sum),
        foc_signed=float(cam.max() / (np.abs(cam).mean() + 1e-12)),
        foc_abs=float(cam_abs.max() / (cam_abs.mean() + 1e-12)),
        last_k_mass=float(cam_abs[-last_k:].sum() / a_sum) if T >= last_k else float(cam_abs.sum() / a_sum),
        cam_peak_idx=int(np.argmax(cam_abs)),
        cam_peak_value=float(cam_abs.max()),
    )


# ═══════════════════════════════════════════════════════════════════
# 5) Permutation tests
# ═══════════════════════════════════════════════════════════════════


def paired_permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = 2000):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    diff = a - b
    obs = float(diff.mean())
    es = float(obs / (diff.std() + 1e-12))
    cnt = sum(1 for _ in range(n_perm) if abs((diff * np.random.choice([-1, 1], len(a))).mean()) >= abs(obs))
    return obs, cnt / n_perm, es


def cluster_permutation_test_1d(ts_a, ts_b, threshold=2.0, n_perm=1000):
    if sp_stats is None or nd_label is None:
        return None, float("nan"), float("nan")
    a, b = np.asarray(ts_a, np.float64), np.asarray(ts_b, np.float64)
    if a.shape[0] < 3:
        return None, float("nan"), float("nan")
    t_obs, _ = sp_stats.ttest_rel(a, b, axis=0)

    def max_mass(t_arr):
        mask = np.abs(t_arr) > threshold
        if not mask.any():
            return 0.0
        lab, nlab = nd_label(mask)
        return max(float(np.abs(t_arr[lab == i]).sum()) for i in range(1, nlab + 1))

    obs_mass = max_mass(t_obs)
    diff = a - b
    n = a.shape[0]
    null = []
    for _ in range(n_perm):
        s = np.random.choice([-1, 1], size=(n, 1))
        t_ = np.mean(diff * s, axis=0) / (np.std(diff * s, axis=0) / np.sqrt(n) + 1e-12)
        null.append(max_mass(t_))
    p = (np.sum(np.array(null) >= obs_mass) + 1) / (n_perm + 1)
    return t_obs, float(p), float(obs_mass)


# ═══════════════════════════════════════════════════════════════════
# 6) Deletion test
# ═══════════════════════════════════════════════════════════════════


def deletion_test_both(
    model: nn.Module,
    X_batch: torch.Tensor,
    cam_batch: np.ndarray,
    mask_percentile: int = 90,
    random_trials: int = 100,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    X_batch = X_batch.to(device)
    with torch.no_grad():
        y0 = model(X_batch).squeeze(-1).cpu().numpy()
    mean_feat = X_batch.mean(dim=[0, 1], keepdim=True)
    N, T, F_ = X_batch.shape

    Xm = X_batch.clone()
    for i in range(N):
        cam = np.asarray(cam_batch[i]).ravel()
        thr = np.percentile(cam, mask_percentile)
        Xm[i, cam >= thr, :] = mean_feat[0, 0, :]
    with torch.no_grad():
        y_imp = model(Xm).squeeze(-1).cpu().numpy()
    imp_delta = np.abs(y0 - y_imp)

    n_masked = max(1, int(T * (100 - mask_percentile) / 100))
    rand_deltas = []
    for _ in range(random_trials):
        Xr = X_batch.clone()
        for i in range(N):
            ridx = np.random.choice(T, size=n_masked, replace=False)
            Xr[i, ridx, :] = mean_feat[0, 0, :]
        with torch.no_grad():
            y_r = model(Xr).squeeze(-1).cpu().numpy()
        rand_deltas.append(np.abs(y0 - y_r))
    rand_deltas = np.stack(rand_deltas, axis=0)
    return imp_delta, rand_deltas.mean(axis=0), rand_deltas.std(axis=0)


# ═══════════════════════════════════════════════════════════════════
# 7) Feature extraction helpers
# ═══════════════════════════════════════════════════════════════════


def extract_raw_stats(X_np: np.ndarray) -> np.ndarray:
    last = X_np[:, -1, :]
    prev = X_np[:, -2, :] if X_np.shape[1] > 1 else np.zeros_like(last)
    return np.hstack([last, last - prev])


def get_model_embeddings(model: nn.Module, dl, device) -> np.ndarray:
    first_linear = None
    for m in model.head:
        if isinstance(m, nn.Linear):
            first_linear = m
            break
    if first_linear is None:
        raise RuntimeError("head Linear not found")
    buf = []
    def hook_fn(module, inp, out):
        buf.append(inp[0].detach().cpu().numpy())
    h = first_linear.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        for x, _ in dl:
            _ = model(x.to(device))
    h.remove()
    return np.vstack(buf)


# ═══════════════════════════════════════════════════════════════════
# 8) Grad-CAM engine (signed variant)
# ═══════════════════════════════════════════════════════════════════


def _find_last_conv_per_branch(model: nn.Module) -> Dict[str, nn.Module]:
    out = {}
    branches = getattr(model, "tcns", None) or getattr(model, "cnns", None)
    if branches is None:
        raise RuntimeError("No tcns/cnns found on model")
    attr_name = "tcns" if hasattr(model, "tcns") else "cnns"
    for i, branch in enumerate(branches):
        last_name, last_mod = None, None
        for name, m in branch.named_modules():
            if isinstance(m, nn.Conv1d):
                last_name, last_mod = name, m
        if last_mod is not None:
            out[f"{attr_name}.{i}.{last_name}"] = last_mod
    if not out:
        raise RuntimeError("Conv1d not found")
    return out


class GradCAMEngine:
    """Computes both signed and abs CAM."""

    def __init__(self, model: nn.Module, device):
        self.model = model
        self.device = device
        self.target_map = _find_last_conv_per_branch(model)
        self.fmap, self.grad = {}, {}
        self._handles = []
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
            try: h.remove()
            except: pass
        self._handles.clear()

    def _cam_1layer_signed(self, fmap, grad, T):
        A, G = fmap[0], grad[0]
        w = G.mean(dim=-1)
        cam = (w.unsqueeze(-1) * A).sum(dim=0).detach().cpu().numpy()
        if cam.size != T:
            cam = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, cam.size), cam)
        return cam

    def generate(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        assert x.dim() == 3 and x.size(0) == 1
        x = x.to(self.device).clone().requires_grad_(True)
        T = x.size(1)
        was = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        y = self.model(x).squeeze()
        y.backward()
        cams_signed = []
        for name in self.target_map:
            if name in self.fmap and name in self.grad:
                cams_signed.append(self._cam_1layer_signed(self.fmap[name], self.grad[name], T))
        self.model.train(was)
        if not cams_signed:
            return np.zeros(T), np.zeros(T)
        stack = np.stack(cams_signed, axis=0)
        cam_signed = stack.mean(axis=0)
        cam_abs = np.abs(cam_signed)
        sm = np.abs(cam_signed).max()
        if sm > 0: cam_signed = cam_signed / sm
        am = cam_abs.max()
        if am > 0: cam_abs = cam_abs / am
        return cam_signed.astype(np.float32), cam_abs.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# 9) Plotting
# ═══════════════════════════════════════════════════════════════════


def _save_fig(fig, path, show=False):
    _ensure_dir(path)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    if show: plt.show()
    else: plt.close(fig)


def plot_mean_cam(cams_e, cams_c, t_curve=None, title="", savepath=None, show=False):
    fig, ax = plt.subplots(figsize=(10, 4))
    m1, m2 = cams_e.mean(0), cams_c.mean(0)
    se1 = cams_e.std(0) / np.sqrt(max(len(cams_e), 1))
    se2 = cams_c.std(0) / np.sqrt(max(len(cams_c), 1))
    x = np.arange(len(m1))
    ax.plot(x, m1, label="event", color="tab:red")
    ax.fill_between(x, m1 - se1, m1 + se1, alpha=0.15, color="tab:red")
    ax.plot(x, m2, label="control", color="tab:blue")
    ax.fill_between(x, m2 - se2, m2 + se2, alpha=0.15, color="tab:blue")
    if t_curve is not None:
        for i in np.where(np.abs(t_curve) > 2.0)[0]:
            ax.axvline(i, color="gray", alpha=0.15, lw=3)
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.2); plt.tight_layout()
    if savepath: _save_fig(fig, savepath, show)
    return fig


def plot_distribution_comparison(vals_e, vals_c, metric_name, savepath=None, show=False):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals_e, bins=20, alpha=0.5, label="event", color="tab:red", density=True)
    ax.hist(vals_c, bins=20, alpha=0.5, label="control", color="tab:blue", density=True)
    ax.axvline(vals_e.mean(), color="tab:red", ls="--")
    ax.axvline(vals_c.mean(), color="tab:blue", ls="--")
    ax.set_title(f"{metric_name} distribution"); ax.legend(); ax.grid(alpha=0.2); plt.tight_layout()
    if savepath: _save_fig(fig, savepath, show)
    return fig


def plot_deletion_comparison(imp_e, rand_e, imp_c, rand_c, savepath=None, show=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Event\n(important)", "Event\n(random)", "Control\n(important)", "Control\n(random)"]
    data = [imp_e, rand_e, imp_c, rand_c]
    colors = ["tab:red", "lightsalmon", "tab:blue", "lightsteelblue"]
    bp = ax.boxplot(data, patch_artist=True, labels=labels)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    ax.set_ylabel("|Δy|"); ax.set_title("Deletion: Important vs Random"); ax.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    if savepath: _save_fig(fig, savepath, show)
    return fig


def plot_matching_balance(balance_df, savepath=None, show=False):
    fig, ax = plt.subplots(figsize=(10, 5))
    n = len(balance_df)
    y = np.arange(n)
    ax.barh(y - 0.15, balance_df["smd_before"].values, height=0.3, label="Before", color="lightcoral", alpha=0.7)
    ax.barh(y + 0.15, balance_df["smd_after"].values, height=0.3, label="After", color="steelblue", alpha=0.7)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels(balance_df["feature_name"].values, fontsize=7)
    ax.set_xlabel("SMD"); ax.set_title("Matching Balance"); ax.legend(); ax.grid(alpha=0.2, axis="x")
    plt.tight_layout()
    if savepath: _save_fig(fig, savepath, show)
    return fig


# ═══════════════════════════════════════════════════════════════════
# 10) MAIN: collect_test_windows + run_post_hoc_analysis_v2
# ═══════════════════════════════════════════════════════════════════


def collect_test_windows(dl) -> torch.Tensor:
    return torch.cat([x for x, _ in dl], dim=0)


def run_post_hoc_analysis_v2(
    model: nn.Module,
    meta: dict,
    dl_te,
    cfg,
    *,
    target_col_raw: str = "VIX",
    event_horizons=(5,),
    event_quantiles=(0.95,),
    q_control_range=(0.4, 0.6),
    match_methods=("knn", "propensity"),
    n_neighbors=1,
    max_pairs=200,
    min_index_gap=20,
    enforce_nonoverlap=True,
    n_perm=2000,
    random_deletion_trials=100,
    mask_percentile=90,
    fdr_correction=True,
    last_k=5,
    save_dir="outputs/posthoc",
    sanity_dir="outputs/sanity",
    show=False,
    seed=42,
):
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(sanity_dir, exist_ok=True)
    device = next(model.parameters()).device

    # config
    ph_config = dict(
        event_horizons=list(event_horizons), event_quantiles=list(event_quantiles),
        q_control_range=list(q_control_range), match_methods=list(match_methods),
        n_neighbors=n_neighbors, max_pairs=max_pairs, min_index_gap=min_index_gap,
        enforce_nonoverlap=enforce_nonoverlap, n_perm=n_perm,
        random_deletion_trials=random_deletion_trials, mask_percentile=mask_percentile,
        fdr_correction=fdr_correction, last_k=last_k, seed=seed,
    )
    with open(os.path.join(save_dir, "posthoc_config.json"), "w") as f:
        json.dump(ph_config, f, indent=2, ensure_ascii=False)

    # collect
    df_te = meta["df_te"]
    target_mode = str(meta["target_mode"]).lower()
    seq_len = cfg.seq_len
    X_scaled = collect_test_windows(dl_te)
    valid_dates = df_te.index[seq_len:]
    n = min(len(valid_dates), len(X_scaled))
    valid_dates, X_scaled = valid_dates[:n], X_scaled[:n]

    if target_col_raw in df_te.columns:
        level_valid = df_te[target_col_raw].values[seq_len:seq_len + n]
    elif target_mode == "log" and meta["model_target_col"] in df_te.columns:
        level_valid = np.exp(df_te[meta["model_target_col"]].values)[seq_len:seq_len + n]
    else:
        raise ValueError("Level series not found for post-hoc event definition")
    dates_arr = np.array(valid_dates)

    # event definition
    all_event_dfs = []
    for h in event_horizons:
        for q in event_quantiles:
            all_event_dfs.append(build_event_definition_df(level_valid, dates_arr, h, q, q_control_range))
    pd.concat(all_event_dfs, ignore_index=True).to_csv(os.path.join(save_dir, "event_definition.csv"), index=False)

    main_edf = all_event_dfs[0]
    ev_idx_all = np.where(main_edf["event_label"].values.astype(bool))[0]
    ct_idx_all = np.where(main_edf["control_label"].values.astype(bool))[0]
    if ev_idx_all.size < 3 or ct_idx_all.size < 3:
        print("[PostHoc] Not enough samples. Aborting."); return None

    if enforce_nonoverlap:
        ev_idx = subsample_nonoverlap(ev_idx_all, min_index_gap)
        ct_idx = subsample_nonoverlap(ct_idx_all, min_index_gap)
    else:
        ev_idx, ct_idx = ev_idx_all, ct_idx_all
    if len(ev_idx) < 3 or len(ct_idx) < 3:
        print("[PostHoc] Not enough after subsampling. Aborting."); return None

    pd.DataFrame({
        "date": dates_arr, "index_in_test": np.arange(n),
        "used_in_analysis": np.isin(np.arange(n), np.concatenate([ev_idx, ct_idx])).astype(int),
        "used_as_event": np.isin(np.arange(n), ev_idx).astype(int),
        "used_as_control": np.isin(np.arange(n), ct_idx).astype(int),
    }).to_csv(os.path.join(save_dir, "analysis_indices.csv"), index=False)

    # matching
    raw_feats = extract_raw_stats(X_scaled.numpy())
    try: emb_feats = get_model_embeddings(model, dl_te, device)[:n]
    except: emb_feats = raw_feats

    all_pairs_dfs, all_balance_dfs = [], []
    main_pairs, main_method = None, None
    for mm in match_methods:
        feats = emb_feats if mm == "embedding" else raw_feats
        pm = "propensity" if mm == "propensity" else "knn"
        pairs, bal_df = match_and_report(ev_idx, ct_idx, feats, method=pm, k_neighbors=n_neighbors)
        bal_df["match_method"] = mm
        all_balance_dfs.append(bal_df)
        if max_pairs and len(pairs) > max_pairs: pairs = pairs[:max_pairs]
        for pid, (ei, cis, dist) in enumerate(pairs):
            all_pairs_dfs.append({"pair_id": pid, "event_idx": ei, "event_date": str(dates_arr[ei]),
                                  "control_idx": cis[0], "control_date": str(dates_arr[cis[0]]),
                                  "match_method": mm, "distance_in_match_space": dist})
        if main_pairs is None:
            main_pairs, main_method = pairs, mm

    pd.DataFrame(all_pairs_dfs).to_csv(os.path.join(save_dir, "matched_pairs.csv"), index=False)
    balance_df = pd.concat(all_balance_dfs, ignore_index=True)
    balance_df.to_csv(os.path.join(save_dir, "matching_balance.csv"), index=False)
    plot_matching_balance(all_balance_dfs[0], os.path.join(save_dir, "matching_balance_plot.png"), show)

    # CAM
    me = np.array([p[0] for p in main_pairs], dtype=int)
    mc = np.array([p[1][0] for p in main_pairs], dtype=int)
    cam_engine = GradCAMEngine(model, device)
    cams_e_s, cams_e_a, cams_c_s, cams_c_a = [], [], [], []
    for i in me:
        cs, ca = cam_engine.generate(X_scaled[i:i+1]); cams_e_s.append(cs); cams_e_a.append(ca)
    for i in mc:
        cs, ca = cam_engine.generate(X_scaled[i:i+1]); cams_c_s.append(cs); cams_c_a.append(ca)
    cam_engine.remove_hooks()
    cams_e_s, cams_e_a = np.array(cams_e_s), np.array(cams_e_a)
    cams_c_s, cams_c_a = np.array(cams_c_s), np.array(cams_c_a)

    # CAM summary
    cam_rows = []
    for pid, (ei, cis, _) in enumerate(main_pairs):
        cam_rows.append({"pair_id": pid, "group": "event", "date": str(dates_arr[ei]), **_cam_stats(cams_e_s[pid], last_k)})
        cam_rows.append({"pair_id": pid, "group": "control", "date": str(dates_arr[cis[0]]), **_cam_stats(cams_c_s[pid], last_k)})
    cam_summary_df = pd.DataFrame(cam_rows)
    cam_summary_df.to_csv(os.path.join(save_dir, "cam_summary_per_sample.csv"), index=False)

    # tests
    test_rows = []
    ev_df = cam_summary_df[cam_summary_df["group"] == "event"]
    ct_df = cam_summary_df[cam_summary_df["group"] == "control"]
    for variant, ce, cc in [("abs", cams_e_a, cams_c_a), ("signed", cams_e_s, cams_c_s)]:
        for metric in ["com_abs", "com_signed", "foc_abs", "foc_signed", "last_k_mass"]:
            if metric.endswith("abs") and variant != "abs": continue
            if metric.endswith("signed") and variant != "signed": continue
            if metric == "last_k_mass" and variant != "abs": continue
            a, b = ev_df[metric].values, ct_df[metric].values
            stat, p_raw, es = paired_permutation_test(a, b, n_perm)
            test_rows.append({"metric_name": metric, "cam_variant": variant, "statistic": stat,
                              "effect_size": es, "p_raw": p_raw, "p_fdr": np.nan, "n_pairs": len(a)})
        t_curve, p_c, obs_m = cluster_permutation_test_1d(ce, cc, n_perm=min(n_perm, 1000))
        test_rows.append({"metric_name": f"cluster_{variant}", "cam_variant": variant, "statistic": obs_m,
                          "effect_size": np.nan, "p_raw": p_c, "p_fdr": np.nan, "n_pairs": len(ce)})

    test_df = pd.DataFrame(test_rows)
    if fdr_correction and len(test_df) > 0:
        test_df["p_fdr"] = _benjamini_hochberg(np.nan_to_num(test_df["p_raw"].values, nan=1.0))
    test_df.to_csv(os.path.join(save_dir, "posthoc_tests.csv"), index=False)

    # deletion
    imp_e, rand_e, _ = deletion_test_both(model, X_scaled[me].to(device), cams_e_a, mask_percentile, random_deletion_trials, device)
    imp_c, rand_c, _ = deletion_test_both(model, X_scaled[mc].to(device), cams_c_a, mask_percentile, random_deletion_trials, device)
    del_rows = []
    for pid in range(len(me)):
        del_rows.append({"pair_id": pid, "group": "event", "date": str(dates_arr[me[pid]]),
                         "important_deletion_delta": float(imp_e[pid]), "random_deletion_delta_mean": float(rand_e[pid]), "mask_percentile": mask_percentile})
        del_rows.append({"pair_id": pid, "group": "control", "date": str(dates_arr[mc[pid]]),
                         "important_deletion_delta": float(imp_c[pid]), "random_deletion_delta_mean": float(rand_c[pid]), "mask_percentile": mask_percentile})
    pd.DataFrame(del_rows).to_csv(os.path.join(save_dir, "deletion_scores.csv"), index=False)

    # plots
    t_abs = cluster_permutation_test_1d(cams_e_a, cams_c_a, n_perm=500)[0] if sp_stats else None
    plot_mean_cam(cams_e_a, cams_c_a, t_abs, f"Mean CAM (abs) — {len(main_pairs)} pairs",
                  os.path.join(save_dir, "mean_cam_abs.png"), show)
    t_sig = cluster_permutation_test_1d(cams_e_s, cams_c_s, n_perm=500)[0] if sp_stats else None
    plot_mean_cam(cams_e_s, cams_c_s, t_sig, f"Mean CAM (signed) — {len(main_pairs)} pairs",
                  os.path.join(save_dir, "mean_cam_signed.png"), show)
    plot_distribution_comparison(ev_df["com_abs"].values, ct_df["com_abs"].values, "COM (abs)",
                                os.path.join(save_dir, "com_distribution.png"), show)
    plot_distribution_comparison(ev_df["foc_abs"].values, ct_df["foc_abs"].values, "FOC (abs)",
                                os.path.join(save_dir, "foc_distribution.png"), show)
    plot_deletion_comparison(imp_e, rand_e, imp_c, rand_c,
                            os.path.join(save_dir, "deletion_vs_random.png"), show)

    # sanity
    mean_abs_e, mean_abs_c = cams_e_a.mean(0), cams_c_a.mean(0)
    peaks_e = [int(np.argmax(np.abs(c))) for c in cams_e_s]
    peaks_c = [int(np.argmax(np.abs(c))) for c in cams_c_s]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(mean_abs_e, label="event", color="tab:red"); axes[0].plot(mean_abs_c, label="control", color="tab:blue")
    axes[0].set_title("Mean abs CAM"); axes[0].legend(); axes[0].grid(alpha=0.2)
    axes[1].hist(peaks_e, bins=15, alpha=0.5, label="event", color="tab:red")
    axes[1].hist(peaks_c, bins=15, alpha=0.5, label="control", color="tab:blue")
    axes[1].set_title("Peak distribution"); axes[1].legend(); axes[1].grid(alpha=0.2)
    plt.tight_layout()
    _save_fig(fig, os.path.join(sanity_dir, "cam_alignment_sanity.png"), show)

    shift_stat, shift_p, shift_es = paired_permutation_test(np.array(peaks_e, float), np.array(peaks_c, float), n_perm)
    pd.DataFrame([{"metric": "cam_peak_shift", "event_mean": np.mean(peaks_e), "control_mean": np.mean(peaks_c),
                    "diff": shift_stat, "p_value": shift_p, "effect_size": shift_es}]).to_csv(
        os.path.join(sanity_dir, "cam_shift_test.csv"), index=False)

    return {"test_df": test_df, "cam_summary_df": cam_summary_df, "balance_df": balance_df,
            "n_pairs": len(main_pairs), "main_method": main_method, "save_dir": save_dir}
