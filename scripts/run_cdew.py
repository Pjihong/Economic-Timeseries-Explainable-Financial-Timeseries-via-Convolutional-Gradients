#!/usr/bin/env python
"""Run C-DEW analysis and optional concept dashboard."""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import vix_xai as vtrx
from vix_xai.metrics import run_metrics_over_time
import vix_xai.concepts as concepts_mod
from vix_xai.concepts import run_cdew_analysis, run_concept_dashboard

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv-path", default="data/raw/timeseries_data.csv")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--bundle-path", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--save-dir-cdew", default="outputs/cdew")
    p.add_argument("--save-dir-concepts", default="outputs/concepts")
    p.add_argument("--save-dir-metrics-cache", default="outputs/metrics_for_cdew")
    p.add_argument("--run-dashboard", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show", action="store_true")
    p.add_argument("--debug-max-n", type=int, default=None)
    return p.parse_args()

def find_bundle(out_dir, bp):
    if bp: return Path(bp)
    d = Path(out_dir)/"bundles"
    for n in ["best_tcn_bundle.pt","best_model_bundle.pt"]:
        if (d/n).exists(): return d/n
    raise FileNotFoundError(f"No bundle in {d}")

def _flight_to_safety(df_raw, df_ref):
    try:
        gold = concepts_mod._resolve_col(df_raw, ("Gold",))
        spx = concepts_mod._resolve_col(df_raw, ("SPX","S&P","S&P 500"))
        gr, sr = df_raw[gold].pct_change(), df_raw[spx].pct_change()
        thr = np.nanquantile(df_ref[gold].pct_change().dropna().values, 0.90)
        return (gr >= thr) & (sr < 0)
    except: return pd.Series(False, index=df_raw.index)

def _liquidity_squeeze(df_raw, df_ref):
    try:
        dxy = concepts_mod._resolve_col(df_raw, ("DXY",))
        vix = concepts_mod._resolve_col(df_raw, ("VIX",))
        dr, vr = df_raw[dxy].pct_change(), df_raw[vix].pct_change()
        thr = np.nanquantile(df_ref[dxy].pct_change().dropna().values, 0.90)
        return (dr >= thr) & (vr > 0)
    except: return pd.Series(False, index=df_raw.index)

def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else vtrx.get_device()
    bp = find_bundle(args.out_dir, args.bundle_path)
    model, mb, snap = vtrx.load_model_bundle(str(bp), device=device)
    cfg = vtrx.Config(**snap["cfg"]); cfg.csv_path = args.csv_path
    tm = snap.get("target_mode", mb.get("target_mode","level"))
    df_raw = vtrx.load_frame(cfg.csv_path, cfg.index_col, list(cfg.drop_cols))
    _,_,_,meta = vtrx.build_dataloaders(df_raw=df_raw, target_col=cfg.target_col,
        seq_len=cfg.seq_len, batch_size=cfg.batch_size, train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio, num_workers=0, pin_memory=False,
        persistent_workers=False, target_mode=tm)

    met = run_metrics_over_time(model=model, meta=meta, cfg=cfg, df_raw=df_raw,
        device=device, reference_mode="fixed", band=5, horizon=5, q_event=0.98,
        evaluate_auc=False, evaluate_retrieval=False,
        save_dir=args.save_dir_metrics_cache, seed=args.seed, debug_max_n=args.debug_max_n)

    run_cdew_analysis(model=model, meta=meta, cfg=cfg, df_raw=df_raw, device=device,
        Ea_all=met["Ea_all"], cam_all=met["cam_all"], raw_windows=met["raw_windows"],
        is_event=met["is_event"], label_dates=met["label_dates"], N=met["N"], refs=met["refs"],
        safe_asset_cols=("Gold",), risk_asset_col=("SPX","S&P"),
        q_safe=0.90, band=5, n_perm=500, save_dir=args.save_dir_cdew, seed=args.seed)

    if args.run_dashboard:
        run_concept_dashboard(model=model, meta=meta, cfg=cfg, df_raw=df_raw, device=device,
            Ea_all=met["Ea_all"], cam_all=met["cam_all"], raw_windows=met["raw_windows"],
            is_event=met["is_event"], label_dates=met["label_dates"], N=met["N"], refs=met["refs"],
            concept_definitions={"FlightToSafety": _flight_to_safety, "LiquiditySqueeze": _liquidity_squeeze},
            save_dir=args.save_dir_concepts, seed=args.seed)
    print("[DONE]")

if __name__ == "__main__":
    main()
