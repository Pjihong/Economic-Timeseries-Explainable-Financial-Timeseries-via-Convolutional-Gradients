#!/usr/bin/env python
"""Run post-hoc CAM analysis on a saved model bundle."""
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import vix_xai as vtrx
from vix_xai.posthoc import run_post_hoc_analysis_v2

DROP = ["Silver","Copper","USD/GBP","USD/CNY","USD/JPY","USD/EUR","USD/CAD"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv-path", default="data/raw/timeseries_data.csv")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--bundle-path", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--save-dir", default="outputs/posthoc")
    p.add_argument("--sanity-dir", default="outputs/sanity")
    p.add_argument("--n-perm", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show", action="store_true")
    return p.parse_args()

def find_bundle(out_dir, bp):
    if bp: return Path(bp)
    d = Path(out_dir)/"bundles"
    for n in ["best_tcn_bundle.pt","best_model_bundle.pt"]:
        if (d/n).exists(): return d/n
    raise FileNotFoundError(f"No bundle in {d}")

def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else vtrx.get_device()
    bp = find_bundle(args.out_dir, args.bundle_path)
    model, mb, snap = vtrx.load_model_bundle(str(bp), device=device)
    cfg = vtrx.Config(**snap["cfg"]); cfg.csv_path = args.csv_path
    tm = snap.get("target_mode", mb.get("target_mode","level"))
    df_raw = vtrx.load_frame(cfg.csv_path, cfg.index_col, list(cfg.drop_cols))
    _,_,dl_te,meta = vtrx.build_dataloaders(df_raw=df_raw, target_col=cfg.target_col,
        seq_len=cfg.seq_len, batch_size=cfg.batch_size, train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio, num_workers=0, pin_memory=False,
        persistent_workers=False, target_mode=tm)
    result = run_post_hoc_analysis_v2(model=model, meta=meta, dl_te=dl_te, cfg=cfg,
        target_col_raw=cfg.target_col, event_horizons=[5], event_quantiles=[0.95],
        match_methods=["knn"], n_perm=args.n_perm, save_dir=args.save_dir,
        sanity_dir=args.sanity_dir, show=args.show, seed=args.seed)
    print(f"[DONE] pairs={result['n_pairs'] if result else 0}")

if __name__ == "__main__":
    main()
