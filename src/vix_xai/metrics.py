from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from .data import build_dataloaders
from .event_warping import dtdw_1d, dtdw_embedding, dtw_from_cost_matrix, wdtdw_embedding
from .utils import ensure_dir, save_json
from .xai import TimeSeriesGradCAMRegression, collect_test_windows


def _define_events_from_level(level, index, horizon=5, q_event=0.95, q_control_range=(0.4, 0.6)):
    level = np.asarray(level, dtype=np.float64).ravel()
    n = len(level)
    max_changes = np.full(n, np.nan)

    for i in range(n - horizon):
        fut = level[i + 1 : i + 1 + horizon]
        max_changes[i] = np.abs(fut - level[i]).max()

    valid = max_changes[np.isfinite(max_changes)]
    if len(valid) == 0:
        return index[:0], index[:0]

    thr_ev = np.quantile(valid, q_event)
    thr_lo = np.quantile(valid, q_control_range[0])
    thr_hi = np.quantile(valid, q_control_range[1])

    ev_mask = max_changes >= thr_ev
    ct_mask = (max_changes >= thr_lo) & (max_changes <= thr_hi)
    return index[ev_mask], index[ct_mask]


def _auc_ci(y_true, y_score, n_boot=500, alpha=0.05):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan")

    auc = roc_auc_score(y_true, y_score)
    aucs = []
    rng = np.random.default_rng(42)

    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))

    if len(aucs) < 10:
        return float(auc), float("nan"), float("nan")

    lo = np.percentile(aucs, 100 * alpha / 2)
    hi = np.percentile(aucs, 100 * (1 - alpha / 2))
    return float(auc), float(lo), float(hi)


def _precision_recall_at_k(y_true, y_score, k):
    n_pos = int(y_true.sum())
    if n_pos == 0 or k == 0:
        return 0.0, 0.0
    order = np.argsort(-y_score)[:k]
    tp = int(y_true[order].sum())
    return tp / k, tp / n_pos


def select_references(raw_windows, mode="fixed", n_prototypes=5, seed=42):
    N = len(raw_windows)
    means = raw_windows.mean(axis=1)

    if mode == "fixed":
        ref0 = int(np.argmin(np.abs(means - np.median(means))))
        return [{"ref_id": 0, "idx": ref0, "selection_rule": "median_mean", "prototype_cluster": -1}]

    if mode == "multi_ref":
        qs = [0.1, 0.25, 0.5, 0.75, 0.9]
        refs = []
        for i, q in enumerate(qs):
            tgt = np.quantile(means, q)
            idx = int(np.argmin(np.abs(means - tgt)))
            refs.append(
                {
                    "ref_id": i,
                    "idx": idx,
                    "selection_rule": f"quantile_{q:.2f}",
                    "prototype_cluster": -1,
                }
            )
        return refs

    if mode == "prototype":
        feats = raw_windows.reshape(N, -1)
        km = KMeans(n_clusters=min(n_prototypes, N), random_state=seed, n_init=10)
        labels = km.fit_predict(feats)
        refs = []
        for c in range(km.n_clusters):
            members = np.where(labels == c)[0]
            center = km.cluster_centers_[c]
            dists = np.linalg.norm(feats[members] - center, axis=1)
            best = members[np.argmin(dists)]
            refs.append(
                {
                    "ref_id": c,
                    "idx": int(best),
                    "selection_rule": "kmeans_centroid",
                    "prototype_cluster": int(c),
                }
            )
        return refs

    raise ValueError(f"Unknown mode: {mode}")


def extract_timewise_embeddings_batch(model, X_scaled_all, branch_idx, batch_size=256):
    model.eval()
    dev = next(model.parameters()).device
    branches = getattr(model, "tcns", None) or getattr(model, "cnns", None)
    target_block = branches[branch_idx].network[-1]
    buf = {}

    def hook_fn(_, __, out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        buf["out"] = o.detach()

    h = target_block.register_forward_hook(hook_fn)
    outs = []

    try:
        with torch.no_grad():
            for s in range(0, len(X_scaled_all), batch_size):
                xb = X_scaled_all[s : s + batch_size].to(dev)
                _ = model(xb)
                outs.append(buf["out"].transpose(1, 2).cpu().numpy())
    finally:
        h.remove()

    return np.concatenate(outs, axis=0).astype(np.float64)


def compute_cam_all(model, X_scaled_all, device, smooth_steps=0, noise_sigma=0.0):
    model.eval()
    cam_engine = TimeSeriesGradCAMRegression(model, device=device, aggregate="mean")
    cams = []

    try:
        for i in tqdm(range(len(X_scaled_all)), desc="Grad-CAM(all)", unit="win"):
            x = X_scaled_all[i : i + 1].to(device)
            cam, _ = cam_engine.generate(x, smooth_steps=smooth_steps, noise_sigma=noise_sigma)
            cams.append(np.asarray(cam, dtype=np.float64))
    finally:
        cam_engine.remove_hooks()

    return np.stack(cams, axis=0)


def cost_matrix_raw_l1(a, b, band=5):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    T = a.size
    C = np.abs(a[:, None] - b[None, :])
    if band is not None:
        ii = np.arange(T)[:, None]
        jj = np.arange(T)[None, :]
        C = np.where(np.abs(ii - jj) <= band, C, np.inf)
    return C


def run_metrics_over_time(
    *,
    model,
    meta,
    cfg,
    df_raw,
    device,
    reference_mode: str = "fixed",
    n_prototypes: int = 5,
    band: int = 5,
    k_dtdw: int = 3,
    dist_method: str = "wasserstein",
    alpha: float = 1.5,
    top_p: float = 0.20,
    weight_mode: str = "local",
    gamma: float = 1.0,
    horizon: int = 5,
    q_event: float = 0.98,
    evaluate_auc: bool = True,
    evaluate_retrieval: bool = True,
    k_list: List[int] = (1, 5, 10),
    alphas_sensitivity: Optional[List[float]] = None,
    top_ps_sensitivity: Optional[List[float]] = None,
    save_dir: str = "outputs/metrics",
    show: bool = False,
    seed: int = 42,
    debug_max_n: Optional[int] = None,
):
    save_dir = Path(save_dir)
    ensure_dir(save_dir)
    np.random.seed(seed)

    save_json(
        save_dir / "metric_config.json",
        {
            "band": band,
            "k": k_dtdw,
            "dist_method": dist_method,
            "alpha": alpha,
            "top_p": top_p,
            "weight_mode": weight_mode,
            "gamma": gamma,
            "reference_mode": reference_mode,
            "n_prototypes": n_prototypes,
            "horizon": horizon,
            "q_event": q_event,
        },
    )

    df_te = meta["df_te"]
    te_index = df_te.index
    seq_len = cfg.seq_len

    _, _, dl_te, meta_te = build_dataloaders(
        df_raw=df_raw,
        target_col=cfg.target_col,
        seq_len=seq_len,
        batch_size=cfg.batch_size,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.pin_memory and device.type == "cuda"),
        persistent_workers=cfg.persistent_workers,
        target_mode=meta["target_mode"],
    )

    X_scaled_all = collect_test_windows(dl_te).cpu()
    label_dates = np.array(te_index[seq_len:])
    N = min(len(X_scaled_all), len(label_dates))
    if debug_max_n:
        N = min(N, debug_max_n)

    X_scaled_all = X_scaled_all[:N]
    label_dates = label_dates[:N]

    vix_on_te = df_raw.reindex(te_index)[cfg.target_col].to_numpy(dtype=np.float64)
    vix_on_te = np.nan_to_num(vix_on_te)
    raw_windows = np.stack([vix_on_te[i : i + seq_len] for i in range(N)], axis=0)

    level = np.nan_to_num(df_raw.reindex(te_index)[cfg.target_col].to_numpy(dtype=np.float64))
    ev_dates, _ = _define_events_from_level(level, te_index, horizon=horizon, q_event=q_event)
    ev_set = set(pd.to_datetime(ev_dates).tolist())
    is_event = np.array([1 if pd.Timestamp(d) in ev_set else 0 for d in pd.to_datetime(label_dates)], dtype=np.int8)

    refs = select_references(raw_windows, mode=reference_mode, n_prototypes=n_prototypes, seed=seed)
    ref_df = pd.DataFrame(refs)
    ref_df["date"] = [str(label_dates[r["idx"]]) for r in refs]
    ref_df.to_csv(save_dir / "reference_set.csv", index=False)

    feature_names = meta_te["feature_names"]
    branch_idx = feature_names.index(cfg.target_col) if cfg.target_col in feature_names else int(meta_te["target_index"])

    Ea_all = extract_timewise_embeddings_batch(model, X_scaled_all, branch_idx)
    cam_all = compute_cam_all(model, X_scaled_all, device)

    all_rows = []
    for ref_info in refs:
        rid = ref_info["ref_id"]
        j0 = ref_info["idx"]

        for i in tqdm(range(N), desc=f"Metrics ref={rid}", unit="w", leave=False):
            j = j0 if j0 != i else max(0, i - 1)

            C1 = cost_matrix_raw_l1(raw_windows[i], raw_windows[j], band=band)
            res_raw = dtw_from_cost_matrix(C1, band=band, normalize=True)

            res_dtdw = dtdw_1d(
                raw_windows[i],
                raw_windows[j],
                k=k_dtdw,
                method=dist_method,
                band=band,
                normalize=True,
            )

            res_emb = dtdw_embedding(Ea_all[i], Ea_all[j], method="l2", k=0, band=band, normalize=True)

            res_emb_cam = wdtdw_embedding(
                Ea_all[i],
                Ea_all[j],
                g_a=cam_all[i],
                g_b=cam_all[j],
                emb_method="l2",
                emb_k=0,
                band=band,
                normalize=True,
                alpha=alpha,
                top_p=top_p,
                weight_mode=weight_mode,
                gamma=gamma,
            )

            all_rows.append(
                {
                    "label_date": pd.Timestamp(label_dates[i]),
                    "idx_a": i,
                    "ref_id": rid,
                    "idx_ref": j,
                    "tail_label": int(is_event[i]),
                    "raw_dtw_norm": float(res_raw.normalized_cost),
                    "dtdw_norm": float(res_dtdw.normalized_cost),
                    "emb_dtw_norm": float(res_emb.normalized_cost),
                    "emb_cam_dtw_norm": float(res_emb_cam.normalized_cost),
                }
            )

    df_metrics = pd.DataFrame(all_rows).sort_values(["ref_id", "label_date"]).reset_index(drop=True)
    df_metrics.to_csv(save_dir / "dtw_metrics_over_time.csv", index=False)

    metric_cols = ["raw_dtw_norm", "dtdw_norm", "emb_dtw_norm", "emb_cam_dtw_norm"]

    auc_rows = []
    if evaluate_auc:
        for rid in ref_df["ref_id"]:
            sub = df_metrics[df_metrics["ref_id"] == rid]
            y = sub["tail_label"].values
            for mc in metric_cols:
                auc, lo, hi = _auc_ci(y, sub[mc].values)
                auc_rows.append(
                    {
                        "metric_name": mc,
                        "label_name": "tail_event",
                        "auc": auc,
                        "ci_low": lo,
                        "ci_high": hi,
                        "reference_mode": reference_mode,
                        "ref_id": rid,
                    }
                )
    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(save_dir / "metric_auc.csv", index=False)

    ret_rows = []
    if evaluate_retrieval:
        for rid in ref_df["ref_id"]:
            sub = df_metrics[df_metrics["ref_id"] == rid]
            y = sub["tail_label"].values
            for mc in metric_cols:
                scores = sub[mc].values
                for kk in k_list:
                    p, r = _precision_recall_at_k(y, scores, kk)
                    ret_rows.append(
                        {
                            "metric_name": mc,
                            "k": kk,
                            "precision_at_k": p,
                            "recall_at_k": r,
                            "event_query_only": False,
                            "reference_mode": reference_mode,
                            "ref_id": rid,
                        }
                    )
    ret_df = pd.DataFrame(ret_rows)
    ret_df.to_csv(save_dir / "retrieval_at_k.csv", index=False)

    sens_rows = []
    if len(refs) > 1:
        for rid in ref_df["ref_id"]:
            sub = df_metrics[df_metrics["ref_id"] == rid]
            y = sub["tail_label"].values
            for mc in metric_cols:
                auc_val = _auc_ci(y, sub[mc].values)[0]
                ev_mean = float(sub.loc[sub["tail_label"] == 1, mc].mean()) if sub["tail_label"].sum() > 0 else float("nan")
                ne_mean = float(sub.loc[sub["tail_label"] == 0, mc].mean())
                sens_rows.append(
                    {
                        "metric_name": mc,
                        "ref_id": rid,
                        "auc": auc_val,
                        "mean_distance_event": ev_mean,
                        "mean_distance_nonevent": ne_mean,
                    }
                )
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(save_dir / "ref_sensitivity.csv", index=False)

    alphas_s = alphas_sensitivity or [0.5, 1.0, 1.5, 2.0, 3.0]
    top_ps_s = top_ps_sensitivity or [0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    atp_rows = []

    ref0_idx = refs[0]["idx"]
    sub_idx = np.random.choice(N, size=min(N, 200), replace=False)
    y_sub = is_event[sub_idx]

    for a_val in tqdm(alphas_s, desc="alpha sens"):
        for tp_val in top_ps_s:
            scores = []
            for si in sub_idx:
                j = ref0_idx if ref0_idx != si else max(0, si - 1)
                res = wdtdw_embedding(
                    Ea_all[si],
                    Ea_all[j],
                    g_a=cam_all[si],
                    g_b=cam_all[j],
                    emb_method="l2",
                    emb_k=0,
                    band=band,
                    normalize=True,
                    alpha=a_val,
                    top_p=tp_val,
                    weight_mode=weight_mode,
                    gamma=gamma,
                )
                scores.append(res.normalized_cost)

            scores = np.array(scores)
            auc_val = _auc_ci(y_sub, scores)[0]
            p5, _ = _precision_recall_at_k(y_sub, scores, 5)
            atp_rows.append(
                {
                    "metric_name": "emb_cam_dtw_norm",
                    "alpha": a_val,
                    "top_p": tp_val,
                    "auc": auc_val,
                    "precision_at_5": p5,
                    "selected_on_val": (a_val == alpha and tp_val == top_p),
                }
            )

    atp_df = pd.DataFrame(atp_rows)
    atp_df.to_csv(save_dir / "alpha_top_p_sensitivity.csv", index=False)

    sub0 = df_metrics[df_metrics["ref_id"] == refs[0]["ref_id"]]

    fig, ax = plt.subplots(figsize=(14, 5))
    for mc, lbl in zip(metric_cols, ["RAW DTW", "DTDW", "EMB DTW", "EMB+CAM DTW"]):
        ax.plot(sub0["label_date"], sub0[mc], label=lbl, alpha=0.8, lw=1)
    ax.set_title(f"DTW metrics over time ({reference_mode} ref)")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(save_dir / "metric_over_time.png", dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

    fig, axes = plt.subplots(1, len(metric_cols), figsize=(4 * len(metric_cols), 4), sharey=False)
    for ax, mc in zip(axes, metric_cols):
        x0 = sub0.loc[sub0["tail_label"] == 0, mc]
        x1 = sub0.loc[sub0["tail_label"] == 1, mc]
        ax.boxplot([x0, x1], labels=["non-event", "event"])
        ax.set_title(mc, fontsize=9)
        ax.grid(alpha=0.2, axis="y")
    plt.suptitle("Metric distributions by event label")
    plt.tight_layout()
    fig.savefig(save_dir / "metric_boxplots.png", dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

    if len(auc_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        auc_mean = auc_df.groupby("metric_name")["auc"].mean()
        auc_mean.plot.bar(ax=ax, color="steelblue", edgecolor="black")
        ax.set_ylabel("AUC")
        ax.set_title("Event detection AUC by metric")
        ax.grid(alpha=0.2, axis="y")
        plt.tight_layout()
        fig.savefig(save_dir / "metric_auc_bar.png", dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)

    if len(sens_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        for mc in metric_cols:
            s = sens_df[sens_df["metric_name"] == mc]
            ax.plot(s["ref_id"], s["auc"], marker="o", label=mc)
        ax.set_xlabel("ref_id")
        ax.set_ylabel("AUC")
        ax.set_title("AUC by reference")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        fig.savefig(save_dir / "ref_sensitivity.png", dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)

    if len(ret_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        for mc in metric_cols:
            s = ret_df[(ret_df["metric_name"] == mc) & (ret_df["ref_id"] == refs[0]["ref_id"])]
            ax.plot(s["k"], s["precision_at_k"], marker="o", label=mc)
        ax.set_xlabel("k")
        ax.set_ylabel("Precision@k")
        ax.set_title("Retrieval precision")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        fig.savefig(save_dir / "retrieval_at_k.png", dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return {
        "df_metrics": df_metrics,
        "auc_df": auc_df,
        "ret_df": ret_df,
        "sens_df": sens_df,
        "atp_df": atp_df,
        "Ea_all": Ea_all,
        "cam_all": cam_all,
        "raw_windows": raw_windows,
        "is_event": is_event,
        "label_dates": label_dates,
        "N": N,
        "refs": refs,
    }


run_metrics_over_time_v2 = run_metrics_over_time
