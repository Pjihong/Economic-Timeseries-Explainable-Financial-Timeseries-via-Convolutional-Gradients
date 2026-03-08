#!/usr/bin/env python
"""Run DTW metrics-over-time analysis without depending on src/vix_xai/metrics.py."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import vix_xai as vtrx
import vix_xai.event_wraping as ew
from vix_xai.config import Config, get_device
from vix_xai.models import CNNEnsemble, TCNEnsemble
from vix_xai.xai import collect_test_windows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv-path", default="data/raw/timeseries_data.csv")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--bundle-path", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--reference-mode", default="fixed")
    p.add_argument("--save-dir", default="outputs/metrics")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show", action="store_true")
    p.add_argument("--debug-max-n", type=int, default=None)
    return p.parse_args()


def find_bundle(out_dir, bp):
    if bp:
        return Path(bp)
    d = Path(out_dir) / "bundles"
    for n in ["best_tcn_bundle.pt", "best_model_bundle.pt"]:
        if (d / n).exists():
            return d / n
    raise FileNotFoundError(f"No bundle in {d}")


def _build_model_from_snapshot(snapshot: dict, state_dict: dict, device: torch.device) -> nn.Module:
    cfg_obj = snapshot["cfg"]
    cfg = Config(**cfg_obj) if isinstance(cfg_obj, dict) else cfg_obj
    num_features = snapshot["num_features"]
    target_idx = snapshot["target_idx"]
    out_act = snapshot["out_act"]
    arch = snapshot["arch"]

    if arch == "tcn":
        model = TCNEnsemble(num_features, target_idx, cfg)
    elif arch == "cnn":
        model = CNNEnsemble(num_features, target_idx, cfg)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    if out_act == "softplus":
        model.head.add_module("softplus", nn.Softplus())

    model.load_state_dict(state_dict)
    return model.to(device)


def load_model_bundle_compat(path: str, device: Optional[torch.device] = None):
    device = device or get_device()
    bundle = torch.load(path, map_location=device, weights_only=False)
    model = _build_model_from_snapshot(bundle["snapshot"], bundle["state_dict"], device)
    model.eval()
    return model, bundle.get("meta", {}), bundle["snapshot"]


def _ensure(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _define_events_from_level(level, index, horizon=5, q_event=0.95, q_control_range=(0.4, 0.6)):
    level = np.asarray(level, dtype=np.float64).ravel()
    n = len(level)
    mc = np.full(n, np.nan)
    for i in range(n - horizon):
        mc[i] = np.abs(level[i + 1 : i + 1 + horizon] - level[i]).max()
    valid = mc[np.isfinite(mc)]
    if len(valid) == 0:
        return index[:0], index[:0]
    return (
        index[mc >= np.quantile(valid, q_event)],
        index[(mc >= np.quantile(valid, q_control_range[0])) & (mc <= np.quantile(valid, q_control_range[1]))],
    )


def _auc_ci(y_true, y_score, n_boot=500, alpha=0.05):
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan")
    auc = roc_auc_score(y_true, y_score)
    rng = np.random.default_rng(42)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    if len(aucs) < 10:
        return auc, float("nan"), float("nan")
    return float(auc), float(np.percentile(aucs, 100 * alpha / 2)), float(np.percentile(aucs, 100 * (1 - alpha / 2)))


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
        return [
            {
                "ref_id": i,
                "idx": int(np.argmin(np.abs(means - np.quantile(means, q)))),
                "selection_rule": f"quantile_{q:.2f}",
                "prototype_cluster": -1,
            }
            for i, q in enumerate(qs)
        ]
    if mode == "prototype":
        feats = raw_windows.reshape(N, -1)
        km = KMeans(n_clusters=min(n_prototypes, N), random_state=seed, n_init=10)
        labels = km.fit_predict(feats)
        refs = []
        for c in range(km.n_clusters):
            members = np.where(labels == c)[0]
            dists = np.linalg.norm(feats[members] - km.cluster_centers_[c], axis=1)
            refs.append(
                {
                    "ref_id": c,
                    "idx": int(members[np.argmin(dists)]),
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

    def hook_fn(m, inp, out):
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
    GradCAMClass = getattr(vtrx, "TimeSeriesGradCAMRegression", None)
    if GradCAMClass is None:
        raise AttributeError("TimeSeriesGradCAMRegression not found in vix_xai")
    cam_engine = GradCAMClass(model, device=device, aggregate="mean")
    cams = []
    try:
        for i in tqdm(range(len(X_scaled_all)), desc="Grad-CAM(all)", unit="win"):
            cam, _ = cam_engine.generate(
                X_scaled_all[i : i + 1].to(device),
                smooth_steps=smooth_steps,
                noise_sigma=noise_sigma,
            )
            cams.append(np.asarray(cam, dtype=np.float64))
    finally:
        cam_engine.remove_hooks()
    return np.stack(cams, axis=0)


def cost_matrix_raw_l1(a, b, band=5):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    T = a.size
    C = np.abs(a[:, None] - b[None, :])
    if band is not None:
        ii, jj = np.arange(T)[:, None], np.arange(T)[None, :]
        C = np.where(np.abs(ii - jj) <= band, C, np.inf)
    return C


def run_metrics_over_time_compat(
    *,
    model,
    meta,
    cfg,
    df_raw,
    device,
    reference_mode="fixed",
    n_prototypes=5,
    band=5,
    k_dtdw=3,
    dist_method="wasserstein",
    alpha=1.5,
    top_p=0.20,
    weight_mode="local",
    gamma=1.0,
    horizon=5,
    q_event=0.98,
    evaluate_auc=True,
    evaluate_retrieval=True,
    k_list=(1, 5, 10),
    alphas_sensitivity=None,
    top_ps_sensitivity=None,
    save_dir="outputs/metrics",
    show=False,
    seed=42,
    debug_max_n=None,
):
    _ensure(save_dir)
    np.random.seed(seed)

    with open(os.path.join(save_dir, "metric_config.json"), "w") as f:
        json.dump(
            dict(
                band=band,
                k=k_dtdw,
                dist_method=dist_method,
                alpha=alpha,
                top_p=top_p,
                weight_mode=weight_mode,
                gamma=gamma,
                reference_mode=reference_mode,
                n_prototypes=n_prototypes,
                horizon=horizon,
                q_event=q_event,
            ),
            f,
            indent=2,
        )

    df_te = meta["df_te"]
    te_index = df_te.index
    seq_len = cfg.seq_len

    _, _, dl_te, meta_te = vtrx.build_dataloaders(
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
    X_scaled_all, label_dates = X_scaled_all[:N], label_dates[:N]

    vix_on_te = np.nan_to_num(df_raw.reindex(te_index)[cfg.target_col].to_numpy(np.float64))
    raw_windows = np.stack([vix_on_te[i : i + seq_len] for i in range(N)], axis=0)

    level = np.nan_to_num(df_raw.reindex(te_index)[cfg.target_col].to_numpy(np.float64))
    ev_dates, _ = _define_events_from_level(level, te_index, horizon=horizon, q_event=q_event)
    ev_set = set(pd.to_datetime(ev_dates).tolist())
    is_event = np.array([1 if pd.Timestamp(d) in ev_set else 0 for d in pd.to_datetime(label_dates)], dtype=np.int8)

    refs = select_references(raw_windows, mode=reference_mode, n_prototypes=n_prototypes, seed=seed)
    ref_df = pd.DataFrame(refs)
    ref_df["date"] = [str(label_dates[r["idx"]]) for r in refs]
    ref_df.to_csv(os.path.join(save_dir, "reference_set.csv"), index=False)

    feature_names = meta_te["feature_names"]
    branch_idx = feature_names.index(cfg.target_col) if cfg.target_col in feature_names else int(meta_te["target_index"])
    Ea_all = extract_timewise_embeddings_batch(model, X_scaled_all, branch_idx)
    cam_all = compute_cam_all(model, X_scaled_all, device)

    all_rows = []
    for ref_info in refs:
        rid, j0 = ref_info["ref_id"], ref_info["idx"]
        for i in tqdm(range(N), desc=f"Metrics ref={rid}", unit="w", leave=False):
            j = j0 if j0 != i else max(0, i - 1)
            C1 = cost_matrix_raw_l1(raw_windows[i], raw_windows[j], band=band)
            all_rows.append(
                dict(
                    label_date=pd.Timestamp(label_dates[i]),
                    idx_a=i,
                    ref_id=rid,
                    idx_ref=j,
                    tail_label=int(is_event[i]),
                    raw_dtw_norm=float(ew.dtw_from_cost_matrix(C1, band=band, normalize=True).normalized_cost),
                    dtdw_norm=float(
                        ew.dtdw_1d(
                            raw_windows[i], raw_windows[j], k=k_dtdw, method=dist_method, band=band, normalize=True
                        ).normalized_cost
                    ),
                    emb_dtw_norm=float(
                        ew.dtdw_embedding(Ea_all[i], Ea_all[j], method="l2", k=0, band=band, normalize=True).normalized_cost
                    ),
                    emb_cam_dtw_norm=float(
                        ew.wdtdw_embedding(
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
                        ).normalized_cost
                    ),
                )
            )

    df_metrics = pd.DataFrame(all_rows).sort_values(["ref_id", "label_date"]).reset_index(drop=True)
    df_metrics.to_csv(os.path.join(save_dir, "dtw_metrics_over_time.csv"), index=False)

    metric_cols = ["raw_dtw_norm", "dtdw_norm", "emb_dtw_norm", "emb_cam_dtw_norm"]

    auc_rows = []
    if evaluate_auc:
        for rid in ref_df["ref_id"]:
            sub = df_metrics[df_metrics["ref_id"] == rid]
            y = sub["tail_label"].values
            for mc in metric_cols:
                auc, lo, hi = _auc_ci(y, sub[mc].values)
                auc_rows.append(
                    dict(
                        metric_name=mc,
                        label_name="tail_event",
                        auc=auc,
                        ci_low=lo,
                        ci_high=hi,
                        reference_mode=reference_mode,
                        ref_id=rid,
                    )
                )
    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(os.path.join(save_dir, "metric_auc.csv"), index=False)

    ret_rows = []
    if evaluate_retrieval:
        for rid in ref_df["ref_id"]:
            sub = df_metrics[df_metrics["ref_id"] == rid]
            y = sub["tail_label"].values
            for mc in metric_cols:
                for kk in k_list:
                    p, r = _precision_recall_at_k(y, sub[mc].values, kk)
                    ret_rows.append(dict(metric_name=mc, k=kk, precision_at_k=p, recall_at_k=r, ref_id=rid))
    ret_df = pd.DataFrame(ret_rows)
    ret_df.to_csv(os.path.join(save_dir, "retrieval_at_k.csv"), index=False)

    sens_rows = []
    if len(refs) > 1:
        for rid in ref_df["ref_id"]:
            sub = df_metrics[df_metrics["ref_id"] == rid]
            y = sub["tail_label"].values
            for mc in metric_cols:
                sens_rows.append(
                    dict(
                        metric_name=mc,
                        ref_id=rid,
                        auc=_auc_ci(y, sub[mc].values)[0],
                        mean_distance_event=float(sub.loc[sub["tail_label"] == 1, mc].mean()) if sub["tail_label"].sum() > 0 else float("nan"),
                        mean_distance_nonevent=float(sub.loc[sub["tail_label"] == 0, mc].mean()),
                    )
                )
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(os.path.join(save_dir, "ref_sensitivity.csv"), index=False)

    alphas_s = alphas_sensitivity or [0.5, 1.0, 1.5, 2.0, 3.0]
    top_ps_s = top_ps_sensitivity or [0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    atp_rows = []
    ref0_idx = refs[0]["idx"]
    sub_idx = np.random.choice(N, size=min(N, 200), replace=False)
    y_sub = is_event[sub_idx]
    for a_val in tqdm(alphas_s, desc="alpha sens"):
        for tp_val in top_ps_s:
            scores = np.array(
                [
                    float(
                        ew.wdtdw_embedding(
                            Ea_all[si],
                            Ea_all[ref0_idx if ref0_idx != si else max(0, si - 1)],
                            g_a=cam_all[si],
                            g_b=cam_all[ref0_idx if ref0_idx != si else max(0, si - 1)],
                            emb_method="l2",
                            emb_k=0,
                            band=band,
                            normalize=True,
                            alpha=a_val,
                            top_p=tp_val,
                            weight_mode=weight_mode,
                            gamma=gamma,
                        ).normalized_cost
                    )
                    for si in sub_idx
                ]
            )
            atp_rows.append(
                dict(
                    metric_name="emb_cam_dtw_norm",
                    alpha=a_val,
                    top_p=tp_val,
                    auc=_auc_ci(y_sub, scores)[0],
                    precision_at_5=_precision_recall_at_k(y_sub, scores, 5)[0],
                    selected_on_val=(a_val == alpha and tp_val == top_p),
                )
            )
    atp_df = pd.DataFrame(atp_rows)
    atp_df.to_csv(os.path.join(save_dir, "alpha_top_p_sensitivity.csv"), index=False)

    sub0 = df_metrics[df_metrics["ref_id"] == refs[0]["ref_id"]]
    fig, ax = plt.subplots(figsize=(14, 5))
    for mc, lbl in zip(metric_cols, ["RAW DTW", "DTDW", "EMB DTW", "EMB+CAM DTW"]):
        ax.plot(sub0["label_date"], sub0[mc], label=lbl, alpha=0.8, lw=1)
    ax.set_title(f"DTW metrics ({reference_mode} ref)")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "metric_over_time.png"), dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

    fig, axes = plt.subplots(1, len(metric_cols), figsize=(4 * len(metric_cols), 4), sharey=False)
    for ax, mc in zip(axes, metric_cols):
        ax.boxplot([sub0.loc[sub0["tail_label"] == 0, mc], sub0.loc[sub0["tail_label"] == 1, mc]], labels=["non-event", "event"])
        ax.set_title(mc, fontsize=9)
        ax.grid(alpha=0.2, axis="y")
    plt.suptitle("Metric distributions")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "metric_boxplots.png"), dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

    if len(auc_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        auc_df.groupby("metric_name")["auc"].mean().plot.bar(ax=ax, color="steelblue", edgecolor="black")
        ax.set_ylabel("AUC")
        ax.set_title("AUC by metric")
        ax.grid(alpha=0.2, axis="y")
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "metric_auc_bar.png"), dpi=200)
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
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "ref_sensitivity.png"), dpi=200)
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
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "retrieval_at_k.png"), dpi=200)
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


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else vtrx.get_device()
    bp = find_bundle(args.out_dir, args.bundle_path)
    model, mb, snap = load_model_bundle_compat(str(bp), device=device)
    cfg = vtrx.Config(**snap["cfg"])
    cfg.csv_path = args.csv_path
    tm = snap.get("target_mode", mb.get("target_mode", "level"))
    df_raw = vtrx.load_frame(cfg.csv_path, cfg.index_col, list(cfg.drop_cols))
    _, _, _, meta = vtrx.build_dataloaders(
        df_raw=df_raw,
        target_col=cfg.target_col,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        target_mode=tm,
    )
    result = run_metrics_over_time_compat(
        model=model,
        meta=meta,
        cfg=cfg,
        df_raw=df_raw,
        device=device,
        reference_mode=args.reference_mode,
        band=5,
        k_dtdw=3,
        horizon=5,
        q_event=0.98,
        evaluate_auc=True,
        evaluate_retrieval=True,
        k_list=[1, 5, 10],
        save_dir=args.save_dir,
        show=args.show,
        seed=args.seed,
        debug_max_n=args.debug_max_n,
    )
    print(f"[DONE] N={result['N']}")


if __name__ == "__main__":
    main()