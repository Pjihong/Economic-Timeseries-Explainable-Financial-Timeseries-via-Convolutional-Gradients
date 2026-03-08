from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .event_warping import dtw_from_cost_matrix
from .utils import ensure_dir, save_json


def _norm_colname(s):
    s = str(s).lower()
    s = re.sub(r"[\s\-\_\.\(\)\[\]\/]+", "", s)
    return s.replace("&", "and")


def _resolve_col(df, candidates):
    if isinstance(candidates, str):
        candidates = [candidates]

    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c

    nmap = {_norm_colname(c): c for c in cols}
    for c in candidates:
        key = _norm_colname(c)
        if key in nmap:
            return nmap[key]

    cols_n = [(_norm_colname(c), c) for c in cols]
    for c in candidates:
        hits = [orig for n, orig in cols_n if _norm_colname(c) in n]
        if hits:
            return sorted(hits, key=len)[0]

    raise KeyError(f"Cannot resolve from {candidates}. Available: {cols}")


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals.copy()

    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = np.empty(n, dtype=float)

    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adjusted[order[i]] = prev

    return np.clip(adjusted, 0.0, 1.0)


def _independent_perm(a, b, n_perm=2000):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    obs = float(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n_a = len(a)

    cnt = 0
    for _ in range(n_perm):
        perm = np.random.permutation(pooled)
        stat = perm[:n_a].mean() - perm[n_a:].mean()
        if abs(stat) >= abs(obs):
            cnt += 1

    sa = a.std(ddof=1) if len(a) > 1 else 0.0
    sb = b.std(ddof=1) if len(b) > 1 else 0.0
    pooled_std = np.sqrt(((len(a) - 1) * sa**2 + (len(b) - 1) * sb**2) / max(len(a) + len(b) - 2, 1))
    es = float(obs / (pooled_std + 1e-12))
    return obs, cnt / n_perm, es


def _bootstrap_mean_ci(vals, n_boot=2000, alpha=0.05):
    vals = np.asarray(vals)
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(42)
    means = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n_boot)]
    return (
        float(np.mean(vals)),
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def create_single_concept_labels(
    df_raw,
    train_end_date,
    te_index,
    seq_len,
    N=None,
    safe_asset_cols=("Gold",),
    risk_asset_col=("SPX", "S&P"),
    q_safe=0.90,
    threshold_source="train_only",
):
    te_index = pd.Index(pd.to_datetime(te_index))
    df_full = df_raw.copy()

    safe_resolved = []
    for c in safe_asset_cols:
        try:
            safe_resolved.append(_resolve_col(df_full, c))
        except KeyError:
            pass
    safe_resolved = list(dict.fromkeys(safe_resolved))
    if not safe_resolved:
        raise KeyError(f"No safe_asset_cols found from {safe_asset_cols}")

    risk_col = _resolve_col(df_full, risk_asset_col)

    safe_ret = df_full[safe_resolved].pct_change()
    risk_ret = df_full[risk_col].pct_change()
    safe_score = safe_ret.max(axis=1)

    if threshold_source == "train_only":
        train_mask = df_full.index <= pd.to_datetime(train_end_date)
        train_safe = safe_score[train_mask].dropna().to_numpy(dtype=np.float64)
        thr = float(np.nanquantile(train_safe, q_safe))
    elif threshold_source == "rolling":
        thr = float(np.nanquantile(safe_score.dropna().values, q_safe))
    else:
        thr = float(np.nanquantile(safe_score.to_numpy(dtype=np.float64), q_safe))

    concept_dates = (safe_score >= thr) & (risk_ret < 0)
    concept_on = concept_dates.reindex(te_index).iloc[seq_len:].to_numpy(dtype=np.int8)
    if N is not None:
        concept_on = concept_on[:N]

    info = {
        "safe_score_threshold": thr,
        "risk_condition": f"{risk_col} < 0",
        "source_period": str(train_end_date),
        "threshold_source": threshold_source,
        "safe_cols": safe_resolved,
        "risk_col": risk_col,
        "q_safe": q_safe,
    }
    return concept_on, info


class TCAVExtractorCV:
    def __init__(self, C=1.0, max_iter=5000, cv_folds=5, seed=0):
        self.C = C
        self.max_iter = max_iter
        self.cv_folds = cv_folds
        self.seed = seed
        self.clf_ = None
        self.scaler_ = None
        self._fitted = False
        self.cv_results_ = []
        self.fold_cavs_ = []

    def _pool(self, E):
        E = np.asarray(E, dtype=np.float64)
        return E.mean(axis=1) if E.ndim == 3 else E

    def fit(self, Ea_all, labels):
        X = self._pool(Ea_all)
        y = np.asarray(labels).ravel()
        if len(np.unique(y)) < 2:
            raise ValueError("Need both positive and negative labels for TCAV.")

        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        self.cv_results_ = []
        self.fold_cavs_ = []

        for fold, (tr, va) in enumerate(skf.split(Xs, y)):
            clf = LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                solver="liblinear",
                class_weight="balanced",
                random_state=self.seed,
            )
            clf.fit(Xs[tr], y[tr])

            pred = clf.predict(Xs[va])
            prob = clf.predict_proba(Xs[va])[:, 1]
            acc = accuracy_score(y[va], pred)

            try:
                auc = roc_auc_score(y[va], prob)
            except Exception:
                auc = float("nan")

            pr = float(y[tr].mean())

            w = clf.coef_.ravel() / (self.scaler_.scale_ + 1e-12)
            w = w / (np.linalg.norm(w) + 1e-12)

            self.fold_cavs_.append(w)
            self.cv_results_.append(
                {
                    "fold": fold,
                    "train_size": len(tr),
                    "val_size": len(va),
                    "accuracy": acc,
                    "roc_auc": auc,
                    "positive_rate": pr,
                }
            )

        self.clf_ = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="liblinear",
            class_weight="balanced",
            random_state=self.seed,
        )
        self.clf_.fit(Xs, y)
        self._fitted = True
        return self

    def get_cav(self):
        w = self.clf_.coef_.ravel() / (self.scaler_.scale_ + 1e-12)
        return w / (np.linalg.norm(w) + 1e-12)

    def get_cv_df(self):
        return pd.DataFrame(self.cv_results_)

    def get_stability_df(self):
        rows = []
        cavs = self.fold_cavs_
        for i in range(len(cavs)):
            for j in range(i + 1, len(cavs)):
                cos = float(np.dot(cavs[i], cavs[j]))
                rows.append({"fold_i": i, "fold_j": j, "cosine_similarity": cos})
        return pd.DataFrame(rows)

    def score(self, Ea, labels):
        X = self.scaler_.transform(self._pool(Ea))
        return float(self.clf_.score(X, labels))


def _cost_l1_band(x, y, band=5):
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    T = x.size
    C = np.abs(x[:, None] - y[None, :])
    if band is not None:
        ii = np.arange(T)[:, None]
        jj = np.arange(T)[None, :]
        C = np.where(np.abs(ii - jj) <= band, C, np.inf)
    return C


def compute_cdew_distance(
    E_a,
    E_b,
    g_a,
    g_b,
    v_c,
    band=5,
    normalize_cam=True,
    clip_cam_nonneg=True,
    return_detail=False,
):
    E_a = np.asarray(E_a, np.float64)
    E_b = np.asarray(E_b, np.float64)
    g_a = np.asarray(g_a).ravel()
    g_b = np.asarray(g_b).ravel()
    v_c = np.asarray(v_c).ravel()

    Ca = E_a @ v_c
    Cb = E_b @ v_c

    ga = g_a.copy()
    gb = g_b.copy()

    if clip_cam_nonneg:
        ga = np.clip(ga, 0, None)
        gb = np.clip(gb, 0, None)

    if normalize_cam:
        ga = ga / (np.abs(ga).max() + 1e-12)
        gb = gb / (np.abs(gb).max() + 1e-12)

    Wca = Ca * ga
    Wcb = Cb * gb

    Cm = _cost_l1_band(Wca, Wcb, band=band)
    res = dtw_from_cost_matrix(Cm, band=band, normalize=True)

    if return_detail:
        return float(res.normalized_cost), {"Wca": Wca, "Wcb": Wcb, "Ca": Ca, "Cb": Cb, "ga": ga, "gb": gb}
    return float(res.normalized_cost)


def run_cdew_analysis(
    *,
    model,
    meta,
    cfg,
    df_raw,
    device,
    Ea_all,
    cam_all,
    raw_windows,
    is_event,
    label_dates,
    N,
    refs,
    safe_asset_cols=("Gold",),
    risk_asset_col=("SPX", "S&P"),
    q_safe=0.90,
    threshold_source="train_only",
    tcav_cv_folds=5,
    band=5,
    normalize_cam=True,
    clip_cam_nonneg=True,
    n_perm=2000,
    fdr_correction=True,
    save_dir="outputs/cdew",
    show=False,
    seed=42,
):
    np.random.seed(seed)
    save_dir = Path(save_dir)
    ensure_dir(save_dir)

    save_json(
        save_dir / "cdew_config.json",
        {
            "threshold_source": threshold_source,
            "q_safe": q_safe,
            "safe_asset_cols": list(safe_asset_cols),
            "risk_asset_col": list(risk_asset_col) if isinstance(risk_asset_col, (list, tuple)) else [risk_asset_col],
            "band": band,
            "normalize_cam": normalize_cam,
            "clip_cam_nonneg": clip_cam_nonneg,
            "tcav_cv_folds": tcav_cv_folds,
            "n_perm": n_perm,
        },
    )

    train_end = str(meta["df_tr"].index[-1])
    te_index = meta["df_te"].index

    concept_labels, thr_info = create_single_concept_labels(
        df_raw=df_raw,
        train_end_date=train_end,
        te_index=te_index,
        seq_len=cfg.seq_len,
        N=N,
        safe_asset_cols=safe_asset_cols,
        risk_asset_col=risk_asset_col,
        q_safe=q_safe,
        threshold_source=threshold_source,
    )

    save_json(save_dir / "concept_thresholds.json", thr_info)

    lbl_df = pd.DataFrame(
        {
            "label_date": label_dates[:N],
            "is_event": is_event[:N],
            "is_concept": concept_labels[:N],
            "concept_name": "FlightToSafety",
        }
    )
    lbl_df.to_csv(save_dir / "concept_labels_single.csv", index=False)

    if concept_labels.sum() < 3 or (len(concept_labels) - concept_labels.sum()) < 3:
        return None

    tcav = TCAVExtractorCV(C=1.0, max_iter=5000, cv_folds=tcav_cv_folds, seed=seed)
    tcav.fit(Ea_all[:N], concept_labels)
    v_c = tcav.get_cav()

    cv_df = tcav.get_cv_df()
    cv_df.to_csv(save_dir / "tcav_cv_scores.csv", index=False)

    stab_df = tcav.get_stability_df()
    stab_df.to_csv(save_dir / "cav_stability.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(cv_df["fold"], cv_df["accuracy"], color="steelblue", label="Accuracy")
    ax.bar(cv_df["fold"], cv_df["roc_auc"], alpha=0.5, color="orange", label="AUC")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_title("TCAV CV Scores")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    fig.savefig(save_dir / "tcav_cv_scores.png", dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

    ref0 = refs[0]["idx"]
    cdew_vals = np.zeros(N)

    for i in tqdm(range(N), desc="C-DEW", unit="w"):
        j = ref0 if ref0 != i else max(0, i - 1)
        cdew_vals[i] = compute_cdew_distance(
            Ea_all[i],
            Ea_all[j],
            cam_all[i],
            cam_all[j],
            v_c,
            band=band,
            normalize_cam=normalize_cam,
            clip_cam_nonneg=clip_cam_nonneg,
        )

    cdew_df = pd.DataFrame(
        {
            "label_date": label_dates[:N],
            "is_event": is_event[:N],
            "is_concept": concept_labels[:N],
            "cdew_ref": cdew_vals,
        }
    )
    cdew_df.to_csv(save_dir / "cdew_ref_fixed.csv", index=False)

    effect_rows = []

    ev1 = cdew_vals[is_event[:N] == 1]
    ev0 = cdew_vals[is_event[:N] == 0]
    if len(ev1) > 2 and len(ev0) > 2:
        d, p, es = _independent_perm(ev1, ev0, n_perm)
        effect_rows.append({"effect_name": "event_vs_nonevent", "effect_size": es, "p_raw": p, "p_fdr": np.nan, "n": min(len(ev1), len(ev0))})

    c1 = cdew_vals[concept_labels[:N] == 1]
    c0 = cdew_vals[concept_labels[:N] == 0]
    if len(c1) > 2 and len(c0) > 2:
        d, p, es = _independent_perm(c1, c0, n_perm)
        effect_rows.append({"effect_name": "concept_vs_nonconcept", "effect_size": es, "p_raw": p, "p_fdr": np.nan, "n": min(len(c1), len(c0))})

    ne_mask = is_event[:N] == 0
    c1_ne = cdew_vals[ne_mask & (concept_labels[:N] == 1)]
    c0_ne = cdew_vals[ne_mask & (concept_labels[:N] == 0)]
    if len(c1_ne) > 2 and len(c0_ne) > 2:
        d, p, es = _independent_perm(c1_ne, c0_ne, n_perm)
        effect_rows.append({"effect_name": "concept_controlling_event", "effect_size": es, "p_raw": p, "p_fdr": np.nan, "n": min(len(c1_ne), len(c0_ne))})

    eff_df = pd.DataFrame(effect_rows)
    if fdr_correction and len(eff_df) > 0:
        eff_df["p_fdr"] = _benjamini_hochberg(eff_df["p_raw"].values)
    eff_df.to_csv(save_dir / "cdew_effects.csv", index=False)

    i_ex = min(10, N - 1)
    j_ex = ref0 if ref0 != i_ex else max(0, i_ex - 1)
    _, detail = compute_cdew_distance(
        Ea_all[i_ex],
        Ea_all[j_ex],
        cam_all[i_ex],
        cam_all[j_ex],
        v_c,
        band=band,
        normalize_cam=normalize_cam,
        clip_cam_nonneg=clip_cam_nonneg,
        return_detail=True,
    )

    ex_df = pd.DataFrame(
        {
            "time_idx": np.arange(len(detail["Ca"])),
            "Ca": detail["Ca"],
            "Cb": detail["Cb"],
            "ga": detail["ga"],
            "gb": detail["gb"],
            "Wca": detail["Wca"],
            "Wcb": detail["Wcb"],
        }
    )
    ex_df.to_csv(save_dir / "example_pair_detail.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(cdew_df["label_date"], cdew_df["cdew_ref"], lw=1)
    ax.set_title("C-DEW distance over time")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(save_dir / "cdew_over_time.png", dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([ev0, ev1], labels=["non-event", "event"])
    ax.set_title("C-DEW | event vs non-event")
    ax.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    fig.savefig(save_dir / "cdew_box_event.png", dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([c0, c1], labels=["non-concept", "concept"])
    ax.set_title("C-DEW | concept vs non-concept")
    ax.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    fig.savefig(save_dir / "cdew_box_concept.png", dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(detail["Wca"], label="Wca")
    ax.plot(detail["Wcb"], label="Wcb")
    ax.set_title("Concept-weighted series (example pair)")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(save_dir / "concept_weighted_example_pair.png", dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "cdew_df": cdew_df,
        "effects_df": eff_df,
        "cv_df": cv_df,
        "v_c": v_c,
        "tcav": tcav,
        "concept_labels": concept_labels,
    }


def _create_concept_generic(df_raw, train_end, te_index, seq_len, N, concept_name, pos_condition_fn, threshold_source="train_only"):
    te_index = pd.Index(pd.to_datetime(te_index))
    if threshold_source == "train_only":
        df_train = df_raw[df_raw.index <= pd.to_datetime(train_end)]
        labels_full = pos_condition_fn(df_raw, df_train)
    else:
        labels_full = pos_condition_fn(df_raw, df_raw)

    labels = labels_full.reindex(te_index).iloc[seq_len:].to_numpy(dtype=np.int8)
    if N is not None:
        labels = labels[:N]
    return labels


def run_concept_dashboard(
    *,
    model,
    meta,
    cfg,
    df_raw,
    device,
    Ea_all,
    cam_all,
    raw_windows,
    is_event,
    label_dates,
    N,
    refs,
    concept_definitions: Dict[str, Any],
    threshold_source="train_only",
    tcav_cv_folds=5,
    band=5,
    n_perm=1000,
    save_topk=10,
    save_dir="outputs/concepts",
    show=False,
    seed=42,
):
    np.random.seed(seed)
    save_dir = Path(save_dir)
    ensure_dir(save_dir)

    train_end = str(meta["df_tr"].index[-1])
    te_index = meta["df_te"].index

    label_cols = {"label_date": label_dates[:N], "is_event": is_event[:N]}
    concept_names = []

    for cname, cfn in concept_definitions.items():
        try:
            labels = _create_concept_generic(df_raw, train_end, te_index, cfg.seq_len, N, cname, cfn, threshold_source)
            label_cols[cname] = labels
            concept_names.append(cname)
        except Exception:
            label_cols[cname] = np.zeros(N, dtype=np.int8)
            concept_names.append(cname)

    concept_df = pd.DataFrame(label_cols)
    concept_df.to_csv(save_dir / "concept_labels.csv", index=False)

    prev_rows = []
    for cn in concept_names:
        prev_rows.append(
            {
                "concept": cn,
                "positive_rate": float(concept_df[cn].mean()),
                "positive_count": int(concept_df[cn].sum()),
                "threshold_source": threshold_source,
            }
        )
    pd.DataFrame(prev_rows).to_csv(save_dir / "concept_prevalence.csv", index=False)

    cooc_rows = []
    for i, ci in enumerate(concept_names):
        for j, cj in enumerate(concept_names):
            if j <= i:
                continue
            a = concept_df[ci].values.astype(int)
            b = concept_df[cj].values.astype(int)
            inter = (a & b).sum()
            union = (a | b).sum()
            jaccard = inter / union if union > 0 else 0.0

            n11 = inter
            n00 = ((1 - a) & (1 - b)).sum()
            n10 = (a & (1 - b)).sum()
            n01 = ((1 - a) & b).sum()
            denom = np.sqrt(float((n11 + n10) * (n11 + n01) * (n00 + n10) * (n00 + n01)) + 1e-12)
            phi = (n11 * n00 - n10 * n01) / denom

            cooc_rows.append({"concept_i": ci, "concept_j": cj, "jaccard": jaccard, "phi_corr": phi})

    cooc_df = pd.DataFrame(cooc_rows)
    cooc_df.to_csv(save_dir / "concept_cooccurrence.csv", index=False)

    tcav_rows = []
    concept_cavs = {}

    for cn in concept_names:
        y = concept_df[cn].values
        if y.sum() < 3 or (len(y) - y.sum()) < 3:
            tcav_rows.append(
                {
                    "concept": cn,
                    "cv_acc_mean": np.nan,
                    "cv_acc_std": np.nan,
                    "cv_auc_mean": np.nan,
                    "cv_auc_std": np.nan,
                    "pos_count": int(y.sum()),
                }
            )
            continue

        t = TCAVExtractorCV(cv_folds=tcav_cv_folds, seed=seed)
        t.fit(Ea_all[:N], y)
        cv = t.get_cv_df()
        tcav_rows.append(
            {
                "concept": cn,
                "cv_acc_mean": cv["accuracy"].mean(),
                "cv_acc_std": cv["accuracy"].std(),
                "cv_auc_mean": cv["roc_auc"].mean(),
                "cv_auc_std": cv["roc_auc"].std(),
                "pos_count": int(y.sum()),
            }
        )
        concept_cavs[cn] = t.get_cav()

    pd.DataFrame(tcav_rows).to_csv(save_dir / "tcav_scores_by_concept.csv", index=False)

    ref0 = refs[0]["idx"]
    cdew_cols = {"label_date": label_dates[:N], "is_event": is_event[:N]}

    for cn in concept_names:
        if cn not in concept_cavs:
            cdew_cols[f"cdew_{cn}"] = np.full(N, np.nan)
            continue

        vc = concept_cavs[cn]
        vals = np.zeros(N)

        for i in range(N):
            j = ref0 if ref0 != i else max(0, i - 1)
            vals[i] = compute_cdew_distance(Ea_all[i], Ea_all[j], cam_all[i], cam_all[j], vc, band=band)

        cdew_cols[f"cdew_{cn}"] = vals

    cdew_by_concept = pd.DataFrame(cdew_cols)
    cdew_by_concept.to_csv(save_dir / "cdew_by_concept.csv", index=False)

    cond_rows = []
    for cn in concept_names:
        y_ev = concept_df["is_event"].values
        y_cn = concept_df[cn].values

        mask_on = y_cn == 1
        mask_off = y_cn == 0

        if mask_on.sum() > 0:
            ev_on = y_ev[mask_on]
            m, lo, hi = _bootstrap_mean_ci(ev_on)
        else:
            m, lo, hi = np.nan, np.nan, np.nan

        if mask_off.sum() > 0:
            ev_off = y_ev[mask_off]
            m2, lo2, hi2 = _bootstrap_mean_ci(ev_off)
        else:
            m2, lo2, hi2 = np.nan, np.nan, np.nan

        cond_rows.append(
            {
                "concept": cn,
                "pos": int(mask_on.sum()),
                "p_event_given_concept": m,
                "p_event_given_concept_ci_low": lo,
                "p_event_given_concept_ci_high": hi,
                "p_event_given_nonconcept": m2,
                "p_event_given_nonconcept_ci_low": lo2,
                "p_event_given_nonconcept_ci_high": hi2,
            }
        )

    pd.DataFrame(cond_rows).to_csv(save_dir / "conditional_event_rates.csv", index=False)

    for cn in concept_names:
        col = f"cdew_{cn}"
        if col not in cdew_by_concept.columns:
            continue
        vals = cdew_by_concept[col].values
        if np.all(np.isnan(vals)):
            continue

        topk_idx = np.argsort(-np.nan_to_num(vals))[:save_topk]
        tk_df = pd.DataFrame(
            {
                "rank": np.arange(1, len(topk_idx) + 1),
                "label_date": label_dates[topk_idx],
                "cdew_value": vals[topk_idx],
                "is_event": is_event[topk_idx],
                "concept_on": concept_df[cn].values[topk_idx],
            }
        )
        tk_df.to_csv(save_dir / f"top10_dates_{cn}.csv", index=False)

    n_concepts = len(concept_names)
    fig, axes = plt.subplots(n_concepts + 2, 1, figsize=(14, 3 * (n_concepts + 2)), sharex=True)

    axes[0].fill_between(range(N), 0, is_event[:N], step="mid", alpha=0.3, color="red", label="event")
    axes[0].set_ylabel("Event")
    axes[0].set_title("ConceptDash")
    axes[0].legend(fontsize=8)

    for ci, cn in enumerate(concept_names):
        axes[ci + 1].fill_between(range(N), 0, concept_df[cn].values, step="mid", alpha=0.3, label=cn)
        axes[ci + 1].set_ylabel(cn[:15])
        axes[ci + 1].legend(fontsize=8)

    for cn in concept_names:
        col = f"cdew_{cn}"
        if col in cdew_by_concept.columns:
            axes[-1].plot(cdew_by_concept[col].values, label=cn, alpha=0.7, lw=1)

    axes[-1].set_ylabel("C-DEW")
    axes[-1].set_xlabel("Test index")
    axes[-1].legend(fontsize=7)
    axes[-1].grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(save_dir / "dashboard_main.png", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    for cn in concept_names:
        col = f"cdew_{cn}"
        if col not in cdew_by_concept.columns:
            continue

        vals = cdew_by_concept[col].values
        fig, ax = plt.subplots(figsize=(6, 4))
        v0 = vals[concept_df[cn].values == 0]
        v1 = vals[concept_df[cn].values == 1]
        v0 = v0[np.isfinite(v0)]
        v1 = v1[np.isfinite(v1)]

        if len(v0) > 0 and len(v1) > 0:
            ax.boxplot([v0, v1], labels=[f"non-{cn}", cn])

        ax.set_title(f"C-DEW | {cn}")
        ax.grid(alpha=0.2, axis="y")
        plt.tight_layout()
        fig.savefig(save_dir / f"boxplot_{cn}.png", dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)

    if len(cooc_df) > 0 and len(concept_names) > 1:
        mat = np.eye(len(concept_names))
        name2i = {n: i for i, n in enumerate(concept_names)}

        for _, row in cooc_df.iterrows():
            i = name2i.get(row["concept_i"])
            j = name2i.get(row["concept_j"])
            if i is not None and j is not None:
                mat[i, j] = mat[j, i] = row["phi_corr"]

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(concept_names)))
        ax.set_xticklabels(concept_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(concept_names)))
        ax.set_yticklabels(concept_names, fontsize=8)
        plt.colorbar(im, ax=ax, label="Phi correlation")
        ax.set_title("Concept Co-occurrence")
        plt.tight_layout()
        fig.savefig(save_dir / "concept_cooccurrence_heatmap.png", dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 4))
    for cn in concept_names:
        col = f"cdew_{cn}"
        if col in cdew_by_concept.columns:
            ax.plot(label_dates[:N], cdew_by_concept[col].values, label=cn, alpha=0.7, lw=1)
    ax.set_title("C-DEW over time by concept")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(save_dir / "cdew_over_time_by_concept.png", dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "concept_df": concept_df,
        "cdew_by_concept": cdew_by_concept,
        "concept_cavs": concept_cavs,
        "cooc_df": cooc_df,
    }
