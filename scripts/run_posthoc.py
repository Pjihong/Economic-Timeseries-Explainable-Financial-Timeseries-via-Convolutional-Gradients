#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import vix_tcn_revin_xai_plus as vtrx
import posthoc_analysis_v2 as posthoc_mod


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
    p = argparse.ArgumentParser(description="Run post-hoc analysis on the best saved bundle.")
    p.add_argument("--csv-path", type=str, default="data/raw/timeseries_data.csv")
    p.add_argument("--index-col", type=str, default="날짜")
    p.add_argument("--target-col", type=str, default="VIX")
    p.add_argument("--drop-cols", nargs="*", default=DEFAULT_DROP_COLS)

    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--bundle-path", type=str, default=None)
    p.add_argument("--prefer-tcn", action="store_true")
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--event-horizons", nargs="+", type=int, default=[5, 1])
    p.add_argument("--event-quantiles", nargs="+", type=float, default=[0.95, 0.98])
    p.add_argument("--match-methods", nargs="+", default=["knn", "propensity"])
    p.add_argument("--n-neighbors", type=int, default=1)
    p.add_argument("--max-pairs", type=int, default=200)
    p.add_argument("--min-index-gap", type=int, default=20)
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--random-deletion-trials", type=int, default=100)
    p.add_argument("--mask-percentile", type=int, default=90)
    p.add_argument("--last-k", type=int, default=5)

    p.add_argument("--save-dir", type=str, default="outputs/posthoc")
    p.add_argument("--sanity-dir", type=str, default="outputs/sanity")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_bundle_path(out_dir: str, bundle_path: str | None, prefer_tcn: bool) -> Path:
    if bundle_path is not None:
        path = Path(bundle_path)
        if not path.exists():
            raise FileNotFoundError(f"Bundle not found: {path}")
        return path

    bundles_dir = Path(out_dir) / "bundles"
    candidates = (
        [bundles_dir / "best_tcn_bundle.pt", bundles_dir / "best_model_bundle.pt"]
        if prefer_tcn
        else [bundles_dir / "best_model_bundle.pt", bundles_dir / "best_tcn_bundle.pt"]
    )

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No bundle found in {bundles_dir}. "
        f"Expected one of: {[str(c) for c in candidates]}"
    )


def load_cfg_from_snapshot(snapshot: dict, args: argparse.Namespace) -> vtrx.Config:
    cfg = vtrx.Config(**snapshot["cfg"])
    cfg.csv_path = args.csv_path
    cfg.index_col = args.index_col
    cfg.drop_cols = tuple(args.drop_cols)
    cfg.target_col = args.target_col
    cfg.out_dir = args.out_dir
    return cfg


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    bundle_path = resolve_bundle_path(args.out_dir, args.bundle_path, args.prefer_tcn)

    print(f"[INFO] device      : {device}")
    print(f"[INFO] bundle_path : {bundle_path}")

    model, bundle_meta, snapshot = vtrx.load_model_bundle(str(bundle_path), device=device)
    cfg = load_cfg_from_snapshot(snapshot, args)
    target_mode = snapshot.get("target_mode", bundle_meta.get("target_mode", "level"))

    df_raw = vtrx.load_frame(cfg.csv_path, cfg.index_col, list(cfg.drop_cols))
    print(f"[DATA] shape       : {df_raw.shape}")
    print(f"[DATA] date range  : {df_raw.index[0]} ~ {df_raw.index[-1]}")
    print(f"[MODEL] target_mode: {target_mode}")

    _, _, dl_te, meta = vtrx.build_dataloaders(
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

    result = posthoc_mod.run_post_hoc_analysis_v2(
        model=model,
        meta=meta,
        dl_te=dl_te,
        cfg=cfg,
        target_col_raw=cfg.target_col,
        event_horizons=args.event_horizons,
        event_quantiles=args.event_quantiles,
        enforce_nonoverlap=True,
        min_index_gap=args.min_index_gap,
        match_methods=args.match_methods,
        n_neighbors=args.n_neighbors,
        max_pairs=args.max_pairs,
        n_perm=args.n_perm,
        fdr_correction=True,
        random_deletion_trials=args.random_deletion_trials,
        mask_percentile=args.mask_percentile,
        last_k=args.last_k,
        save_dir=args.save_dir,
        sanity_dir=args.sanity_dir,
        show=args.show,
        seed=args.seed,
    )

    summary = {
        "bundle_path": str(bundle_path),
        "save_dir": args.save_dir,
        "sanity_dir": args.sanity_dir,
        "target_mode": target_mode,
        "n_pairs": None if result is None else int(result["n_pairs"]),
        "main_method": None if result is None else result["main_method"],
    }

    save_path = Path(args.save_dir) / "run_posthoc_summary.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[DONE] summary saved to: {save_path}")


if __name__ == "__main__":
    main()
