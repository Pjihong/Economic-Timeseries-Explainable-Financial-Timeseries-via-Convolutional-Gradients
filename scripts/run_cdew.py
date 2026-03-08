#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import vix_xai as vtrx
from vix_xai.metrics import run_metrics_over_time
import vix_xai.concepts as concepts_mod
from vix_xai.concepts import run_cdew_analysis, run_concept_dashboard


DEFAULT_DROP_COLS = [
    "Silver",
    "Copper",
    "USD/GBP",
    "USD/CNY",
    "USD/JPY",
    "USD/EUR",
    "USD/CAD",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run C-DEW and optional concept dashboard.")
    p.add_argument("--csv-path", type=str, default="data/raw/timeseries_data.csv")
    p.add_argument("--index-col", type=str, default="날짜")
    p.add_argument("--target-col", type=str, default="VIX")
    p.add_argument("--drop-cols", nargs="*", default=DEFAULT_DROP_COLS)

    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--bundle-path", type=str, default=None)
    p.add_argument("--bundle-kind", choices=["auto", "best_model", "best_tcn"], default="auto")
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--reference-mode", type=str, default="multi_ref", choices=["fixed", "multi_ref", "prototype"])
    p.add_argument("--band", type=int, default=5)
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--top-p", type=float, default=0.20)
    p.add_argument("--weight-mode", type=str, default="local")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--q-event", type=float, default=0.98)

    p.add_argument("--q-safe", type=float, default=0.90)
    p.add_argument("--threshold-source", type=str, default="train_only")
    p.add_argument("--tcav-cv-folds", type=int, default=5)
    p.add_argument("--n-perm", type=int, default=2000)

    p.add_argument("--save-dir-cdew", type=str, default="outputs/cdew")
    p.add_argument("--save-dir-concepts", type=str, default="outputs/concepts")
    p.add_argument("--save-dir-metrics-cache", type=str, default="outputs/metrics_for_cdew")
    p.add_argument("--run-dashboard", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show", action="store_true")
    p.add_argument("--debug-max-n", type=int, default=None)
    return p.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_bundle_path(out_dir: str, bundle_path: str | None, bundle_kind: str) -> Path:
    if bundle_path is not None:
        path = Path(bundle_path)
        if not path.exists():
            raise FileNotFoundError(f"Bundle not found: {path}")
        return path

    bundles_dir = Path(out_dir) / "bundles"
    if bundle_kind == "best_model":
        candidates = [bundles_dir / "best_model_bundle.pt"]
    elif bundle_kind == "best_tcn":
        candidates = [bundles_dir / "best_tcn_bundle.pt"]
    else:
        candidates = [bundles_dir / "best_tcn_bundle.pt", bundles_dir / "best_model_bundle.pt"]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"No bundle found in {bundles_dir}")


def load_cfg_from_snapshot(snapshot: dict, args: argparse.Namespace) -> vtrx.Config:
    cfg = vtrx.Config(**snapshot["cfg"])
    cfg.csv_path = args.csv_path
    cfg.index_col = args.index_col
    cfg.drop_cols = tuple(args.drop_cols)
    cfg.target_col = args.target_col
    cfg.out_dir = args.out_dir
    return cfg


def _flight_to_safety(df_raw, df_ref):
    try:
        gold = concepts_mod._resolve_col(df_raw, ("Gold",))
        spx = concepts_mod._resolve_col(df_raw, ("SPX", "S&P", "S&P 500", "SP500"))
        gr = df_raw[gold].pct_change()
        sr = df_raw[spx].pct_change()
        thr = np.nanquantile(df_ref[gold].pct_change().dropna().values, 0.90)
        return (gr >= thr) & (sr < 0)
    except Exception:
        return pd.Series(False, index=df_raw.index)


def _inflation_shock(df_raw, df_ref):
    try:
        wti = concepts_mod._resolve_col(df_raw, ("WTI", "Oil", "CrudeOil", "Crude"))
        gold = concepts_mod._resolve_col(df_raw, ("Gold",))
        wr = df_raw[wti].pct_change()
        gr = df_raw[gold].pct_change()
        thr_w = np.nanquantile(df_ref[wti].pct_change().dropna().values, 0.95)
        return (wr >= thr_w) & (gr > 0)
    except Exception:
        return pd.Series(False, index=df_raw.index)


def _liquidity_squeeze(df_raw, df_ref):
    try:
        dxy = concepts_mod._resolve_col(df_raw, ("DXY", "Dollar", "USD"))
        vix = concepts_mod._resolve_col(df_raw, ("VIX",))
        dr = df_raw[dxy].pct_change()
        vr = df_raw[vix].pct_change()
        thr_d = np.nanquantile(df_ref[dxy].pct_change().dropna().values, 0.90)
        return (dr >= thr_d) & (vr > 0)
    except Exception:
        return pd.Series(False, index=df_raw.index)


def _emerging_market_shock(df_raw, df_ref):
    try:
        kospi = concepts_mod._resolve_col(df_raw, ("KOSPI", "KS11"))
        vix = concepts_mod._resolve_col(df_raw, ("VIX",))
        kr = df_raw[kospi].pct_change()
        vr = df_raw[vix].pct_change()
        thr_k = np.nanquantile(df_ref[kospi].pct_change().dropna().values, 0.05)
        return (kr <= thr_k) & (vr > 0)
    except Exception:
        return pd.Series(False, index=df_raw.index)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    bundle_path = resolve_bundle_path(args.out_dir, args.bundle_path, args.bundle_kind)

    print(f"[INFO] device      : {device}")
    print(f"[INFO] bundle_path : {bundle_path}")

    model, bundle_meta, snapshot = vtrx.load_model_bundle(str(bundle_path), device=device)
    cfg = load_cfg_from_snapshot(snapshot, args)
    target_mode = snapshot.get("target_mode", bundle_meta.get("target_mode", "level"))

    df_raw = vtrx.load_frame(cfg.csv_path, cfg.index_col, list(cfg.drop_cols))
    print(f"[DATA] shape       : {df_raw.shape}")
    print(f"[DATA] date range  : {df_raw.index[0]} ~ {df_raw.index[-1]}")
    print(f"[MODEL] target_mode: {target_mode}")

    _, _, _, meta = vtrx.build_dataloaders(
        df_raw=df_raw,
        target_col=cfg.target_col,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.pin_memory and device.type == "cuda"),
        persistent_workers=cfg.persistent_workers,
        target_mode=target_mode,
    )

    metrics_result = run_metrics_over_time(
        model=model,
        meta=meta,
        cfg=cfg,
        df_raw=df_raw,
        device=device,
        reference_mode=args.reference_mode,
        n_prototypes=5,
        band=args.band,
        k_dtdw=3,
        dist_method="wasserstein",
        alpha=args.alpha,
        top_p=args.top_p,
        weight_mode=args.weight_mode,
        gamma=args.gamma,
        horizon=args.horizon,
        q_event=args.q_event,
        evaluate_auc=False,
        evaluate_retrieval=False,
        k_list=[1, 5, 10],
        alphas_sensitivity=[0.5, 1.0, 1.5, 2.0, 3.0],
        top_ps_sensitivity=[0.10, 0.15, 0.20, 0.30, 0.50, 1.0],
        save_dir=args.save_dir_metrics_cache,
        show=args.show,
        seed=args.seed,
        debug_max_n=args.debug_max_n,
    )

    cdew_result = run_cdew_analysis(
        model=model,
        meta=meta,
        cfg=cfg,
        df_raw=df_raw,
        device=device,
        Ea_all=metrics_result["Ea_all"],
        cam_all=metrics_result["cam_all"],
        raw_windows=metrics_result["raw_windows"],
        is_event=metrics_result["is_event"],
        label_dates=metrics_result["label_dates"],
        N=metrics_result["N"],
        refs=metrics_result["refs"],
        safe_asset_cols=("Gold", "미국채", "US10Y", "10Y", "Treasury"),
        risk_asset_col=("S&P", "S&P 500", "SP500", "SPX"),
        q_safe=args.q_safe,
        threshold_source=args.threshold_source,
        tcav_cv_folds=args.tcav_cv_folds,
        band=args.band,
        normalize_cam=True,
        clip_cam_nonneg=True,
        n_perm=args.n_perm,
        fdr_correction=True,
        save_dir=args.save_dir_cdew,
        show=args.show,
        seed=args.seed,
    )

    dash_result = None
    if args.run_dashboard:
        concept_definitions = {
            "FlightToSafety": _flight_to_safety,
            "InflationShock": _inflation_shock,
            "LiquiditySqueeze": _liquidity_squeeze,
            "EmergingMarketShock": _emerging_market_shock,
        }

        dash_result = run_concept_dashboard(
            model=model,
            meta=meta,
            cfg=cfg,
            df_raw=df_raw,
            device=device,
            Ea_all=metrics_result["Ea_all"],
            cam_all=metrics_result["cam_all"],
            raw_windows=metrics_result["raw_windows"],
            is_event=metrics_result["is_event"],
            label_dates=metrics_result["label_dates"],
            N=metrics_result["N"],
            refs=metrics_result["refs"],
            concept_definitions=concept_definitions,
            threshold_source=args.threshold_source,
            tcav_cv_folds=args.tcav_cv_folds,
            band=args.band,
            n_perm=min(args.n_perm, 1000),
            save_topk=10,
            save_dir=args.save_dir_concepts,
            show=args.show,
            seed=args.seed,
        )

    summary = {
        "bundle_path": str(bundle_path),
        "metrics_cache_dir": args.save_dir_metrics_cache,
        "cdew_dir": args.save_dir_cdew,
        "concepts_dir": args.save_dir_concepts if args.run_dashboard else None,
        "target_mode": target_mode,
        "run_dashboard": bool(args.run_dashboard),
        "cdew_success": cdew_result is not None,
        "dashboard_success": (dash_result is not None) if args.run_dashboard else None,
    }

    save_path = Path(args.save_dir_cdew) / "run_cdew_summary.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[DONE] summary saved to: {save_path}")


if __name__ == "__main__":
    main()
