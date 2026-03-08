#!/usr/bin/env python
"""Train experiment suite and save model bundles."""
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import vix_xai as vtrx


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv-path", default="data/raw/timeseries_data.csv")
    p.add_argument("--index-col", default="날짜")
    p.add_argument("--target-col", default="VIX")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seeds", nargs="+", type=int, default=[5])
    p.add_argument("--quick", action="store_true", help="Quick test run")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else vtrx.get_device()
    cfg = vtrx.Config(
        csv_path=args.csv_path, index_col=args.index_col, target_col=args.target_col,
        drop_cols=("Silver","Copper","USD/GBP","USD/CNY","USD/JPY","USD/EUR","USD/CAD"),
        out_dir=args.out_dir,
        epochs=10 if args.quick else args.epochs,
        patience=5 if args.quick else 20,
        min_epoch=3 if args.quick else 50,
        param_budget=4000, use_amp=(device.type=="cuda"), num_workers=0,
    )
    df_raw = vtrx.load_frame(cfg.csv_path, cfg.index_col, list(cfg.drop_cols))
    print(f"[DATA] {df_raw.shape} cols={list(df_raw.columns)}")

    settings = [("level","mse","none")] if args.quick else [
        ("level","mse","none"),("level","huber","none"),
        ("diff","mse","none"),("log","mse","none"),
    ]
    archs = ["tcn"] if args.quick else ["tcn","cnn"]

    suite = vtrx.run_experiment_suite(cfg, df_raw, experiment_settings=settings,
                                       architectures=archs, seeds=tuple(args.seeds), device=device)
    print(f"\nbest={suite['best_key']} val_rmse={suite['best_val_rmse']:.4f} test_rmse={suite['best_test_rmse']:.4f}")


if __name__ == "__main__":
    main()
