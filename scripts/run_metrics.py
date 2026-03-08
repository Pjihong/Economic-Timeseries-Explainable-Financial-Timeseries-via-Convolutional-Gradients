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
import metrics_over_time_v2 as metrics_mod
import event_warping as ew


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
    p = argparse.ArgumentParser(description="Run metrics-over-time analysis.")
    p.add_argument("--csv-path", type=str, default="data/raw/timeseries_data.csv")
    p.add_argument("--index-col", type=str, default="날짜")
    p.add_argument("--target-col", type=str, default="VIX")
    p.add_argument("--drop-cols", nargs="*", default=DEFAULT_DROP_COLS)

    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--bundle-path", type=str, default=None)
    p.add_argument("--prefer-tcn", action="store_true")
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--reference-mode", type=str, default="multi_ref", choices=["fixed", "multi_ref", "prototype"])
    p.add_argument("--n-prototypes", type=int, default=5)

    p.add_argument("--band", type=int, default=5)
    p.add_argument("--k-dtdw", type=int, default=3)
    p.add_argument("--dist-method", type=str, default="wasserstein")
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--top-p", type=float, default=0.20)
    p.add_argument("--weight-mode", type=str, default="local")
    p.add_argument("--gamma", type=float, default=1.0)

    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--q-event", type=float, default=0.98)

    p.add_argument("--save-dir", type=str, default="outputs/metrics")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show", action="store_true")
    p.add_argument("--debug-max-n", type=int, default=None)
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
    raise FileNotFoundError(f"No bundle found in {bundles_dir}")


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

    result = metrics_mod.run_metrics_over_time_v2(
        vtrx_module=vtrx,
        ew_module=ew,
        model=model,
        meta=meta,
        cfg=cfg,
        df_raw=df_raw,
        device=device,
        reference_mode=args.reference_mode,
        n_prototypes=args.n_prototypes,
        band=args.band,
        k_dtdw=args.k_dtdw,
        dist_method=args.dist_method,
        alpha=args.alpha,
        top_p=args.top_p,
        weight_mode=args.weight_mode,
        gamma=args.gamma,
        horizon=args.horizon,
        q_event=args.q_event,
        evaluate_auc=True,
        evaluate_retrieval=True,
        k_list=[1, 5, 10, 20],
        alphas_sensitivity=[0.5, 1.0, 1.5, 2.0, 3.0],
        top_ps_sensitivity=[0.10, 0.15, 0.20, 0.30, 0.50, 1.0],
        save_dir=args.save_dir,
        show=args.show,
        seed=args.seed,
        debug_max_n=args.debug_max_n,
    )

    summary = {
        "bundle_path": str(bundle_path),
        "save_dir": args.save_dir,
        "target_mode": target_mode,
        "N": int(result["N"]),
        "n_refs": int(len(result["refs"])),
        "reference_mode": args.reference_mode,
    }

    save_path = Path(args.save_dir) / "run_metrics_summary.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[DONE] summary saved to: {save_path}")


if __name__ == "__main__":
    main()
