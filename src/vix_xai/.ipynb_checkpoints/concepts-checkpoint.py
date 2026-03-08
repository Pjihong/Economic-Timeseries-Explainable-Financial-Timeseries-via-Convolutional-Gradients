"""
concepts.py — C-DEW analysis, TCAV with CV, multi-concept dashboard.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .event_wraping import dtw_from_cost_matrix


# ═══════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════


def _ensure(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _norm_colname(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[\s\-\_\.\(\)\[\]\/]+", "", s)
    return s.replace("&", "and")


def _resolve_col(df: pd.DataFrame, candidates) -> str:
    if isinstance(candidates, str):
        candidates = [candidates]
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    nmap = {_norm_colname(c): c for c in cols}
    for c in candidates:
        if _norm_colname(c) in nmap:
            return nmap[_norm_colname(c)]
    cols_n = [(_norm_colname(c), c) for c in cols]
    for c in candidates:
        hits = [orig for n, orig in cols_n if _norm_colname(c) in n]
        if hits:
            return sorted(hits, key=len)[0]
    raise KeyError(f"Cannot resolve from {candidates}. Available: {cols}")


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
    return np.clip(corrected, 0, 1)


def _paired_perm(a, b, n_perm=2000):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    diff = a - b
    obs = diff.mean()
    es = obs / (diff.std() + 1e-12)
    cnt = sum(1 for _ in range(n_perm) if abs((diff * np.random.choice([-1, 1], len(a))).mean()) >= abs(obs))
    return float(obs), cnt / n_perm, float(es)


def _bootstrap_ci(vals, n_boot=2000, alpha=0.05):
    vals = np.asarray(vals)
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(42)
    means = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n_boot)]
    return float(np.mean(vals)), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


# ═══════════════════════════════════════════════════════════════════
# D-1: Concept label creation (train-only threshold)
# ═══════════════════════════════════════════════════════════════════


def create_single_concept_labels(
    df_raw: pd.DataFrame,
    train_end_date: str,
    te_index: pd.Index,
    seq_len: int,
    N: Optional[int] = None,
    safe_asset_cols: Tuple[str, ...] = ("Gold",),
    risk_asset_col: Tuple[str, ...] = ("SPX", "S&P"),
    q_safe: float = 0.90,
    threshold_source: str = "train_only",
) -> Tuple[np.ndarray, dict]:
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
        thr = float(np.nanquantile(safe_score[train_mask].dropna().to_numpy(np.float64), q_safe))
    elif threshold_source == "rolling":
        thr = float(np.nanquantile(safe_score.expanding().quantile(q_safe).reindex(te_index).dropna().values[-1:], 0.5))
    else:
        thr = float(np.nanquantile(safe_score.to_numpy(np.float64), q_safe))

    concept_dates = (safe_score >= thr) & (risk_ret < 0)
    concept_on = concept_dates.reindex(te_index).iloc[seq_len:].to_numpy(dtype=np.int8)
    if N is not None:
        concept_on = concept_on[:N]

    info = dict(safe_score_threshold=thr, risk_condition=f"{risk_col} < 0",
                source_period=str(train_end_date), threshold_source=threshold_source,
                safe_cols=safe_resolved, risk_col=risk_col, q_safe=q_safe)
    return concept_on, info


# ═══════════════════════════════════════════════════════════════════
# D-2: TCAV with CV + CAV stability
# ═══════════════════════════════════════════════════════════════════


class TCAVExtractorCV:
    def __init__(self, C=1.0, max_iter=5000, cv_folds=5, seed=0):
        self.C = C
        self.max_iter = max_iter
        self.cv_folds = cv_folds
        self.seed = seed
        self.clf_ = None
        self.scaler_ = None
        self._fitted = False
        self.cv_results_: list = []
        self.fold_cavs_: list = []

    def _pool(self, E):
        return np.asarray(E, np.float64).mean(axis=1) if E.ndim == 3 else np.asarray(E, np.float64)

    def fit(self, Ea_all, labels):
        X = self._pool(Ea_all)
        y = np.asarray(labels).ravel()
        assert len(np.unique(y)) >= 2
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        self.cv_results_, self.fold_cavs_ = [], []
        for fold, (tr, va) in enumerate(skf.split(Xs, y)):
            clf = LogisticRegression(C=self.C, max_iter=self.max_iter, solver="liblinear",
                                     class_weight="balanced", random_state=self.seed)
            clf.fit(Xs[tr], y[tr])
            pred = clf.predict(Xs[va])
            prob = clf.predict_proba(Xs[va])[:, 1]
            acc = accuracy_score(y[va], pred)
            try: auc = roc_auc_score(y[va], prob)
            except: auc = float("nan")
            w = clf.coef_.ravel() / (self.scaler_.scale_ + 1e-12)
            w = w / (np.linalg.norm(w) + 1e-12)
            self.fold_cavs_.append(w)
            self.cv_results_.append(dict(fold=fold, train_size=len(tr), val_size=len(va),
                                         accuracy=acc, roc_auc=auc, positive_rate=float(y[tr].mean())))
        self.clf_ = LogisticRegression(C=self.C, max_iter=self.max_iter, solver="liblinear",
                                        class_weight="balanced", random_state=self.seed)
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
        for i in range(len(self.fold_cavs_)):
            for j in range(i + 1, len(self.fold_cavs_)):
                rows.append(dict(fold_i=i, fold_j=j, cosine_similarity=float(np.dot(self.fold_cavs_[i], self.fold_cavs_[j]))))
        return pd.DataFrame(rows)

    def score(self, Ea, labels):
        X = self.scaler_.transform(self._pool(Ea))
        return float(self.clf_.score(X, labels))


# ═══════════════════════════════════════════════════════════════════
# D-3: C-DEW distance
# ═══════════════════════════════════════════════════════════════════


def _cost_l1_band(x, y, band=5):
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    T = x.size
    C = np.abs(x[:, None] - y[None, :])
    if band is not None:
        ii, jj = np.arange(T)[:, None], np.arange(T)[None, :]
        C = np.where(np.abs(ii - jj) <= band, C, np.inf)
    return C


def compute_cdew_distance(
    E_a, E_b, g_a, g_b, v_c, band=5,
    normalize_cam=True, clip_cam_nonneg=True, return_detail=False,
):
    E_a, E_b = np.asarray(E_a, np.float64), np.asarray(E_b, np.float64)
    g_a, g_b = np.asarray(g_a).ravel(), np.asarray(g_b).ravel()
    v_c = np.asarray(v_c).ravel()
    Ca, Cb = E_a @ v_c, E_b @ v_c
    ga, gb = g_a.copy(), g_b.copy()
    if clip_cam_nonneg:
        ga, gb = np.clip(ga, 0, None), np.clip(gb, 0, None)
    if normalize_cam:
        ga = ga / (np.abs(ga).max() + 1e-12)
        gb = gb / (np.abs(gb).max() + 1e-12)
    Wca, Wcb = Ca * ga, Cb * gb
    Cm = _cost_l1_band(Wca, Wcb, band=band)
    res = dtw_from_cost_matrix(Cm, band=band, normalize=True)
    if return_detail:
        return float(res.normalized_cost), dict(Wca=Wca, Wcb=Wcb, Ca=Ca, Cb=Cb, ga=ga, gb=gb)
    return float(res.normalized_cost)


# ═══════════════════════════════════════════════════════════════════
# D-MAIN: run_cdew_analysis
# ═══════════════════════════════════════════════════════════════════


def run_cdew_analysis(
    *,
    vtrx_module=None,
    ew_module=None,
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
    _ensure(save_dir)
    np.random.seed(seed)

    with open(os.path.join(save_dir, "cdew_config.json"), "w") as f:
        json.dump(dict(threshold_source=threshold_source, q_safe=q_safe,
                       safe_asset_cols=list(safe_asset_cols),
                       risk_asset_col=list(risk_asset_col) if isinstance(risk_asset_col, (list, tuple)) else [risk_asset_col],
                       band=band, normalize_cam=normalize_cam, clip_cam_nonneg=clip_cam_nonneg,
                       tcav_cv_folds=tcav_cv_folds, n_perm=n_perm), f, indent=2, ensure_ascii=False)

    train_end = str(meta["df_tr"].index[-1])
    te_index = meta["df_te"].index
    concept_labels, thr_info = create_single_concept_labels(
        df_raw, train_end, te_index, cfg.seq_len, N,
        safe_asset_cols=safe_asset_cols, risk_asset_col=risk_asset_col,
        q_safe=q_safe, threshold_source=threshold_source)

    with open(os.path.join(save_dir, "concept_thresholds.json"), "w") as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in thr_info.items()}, f, indent=2)

    pd.DataFrame({"label_date": label_dates[:N], "is_event": is_event[:N],
                   "is_concept": concept_labels[:N], "concept_name": "FlightToSafety"}).to_csv(
        os.path.join(save_dir, "concept_labels_single.csv"), index=False)

    if concept_labels.sum() < 3 or (len(concept_labels) - concept_labels.sum()) < 3:
        print("[C-DEW] Not enough concept samples. Aborting."); return None

    tcav = TCAVExtractorCV(C=1.0, max_iter=5000, cv_folds=tcav_cv_folds, seed=seed)
    tcav.fit(Ea_all[:N], concept_labels)
    v_c = tcav.get_cav()
    tcav.get_cv_df().to_csv(os.path.join(save_dir, "tcav_cv_scores.csv"), index=False)
    tcav.get_stability_df().to_csv(os.path.join(save_dir, "cav_stability.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    cv = tcav.get_cv_df()
    ax.bar(cv["fold"], cv["accuracy"], color="steelblue", label="Accuracy")
    ax.bar(cv["fold"], cv["roc_auc"], alpha=0.5, color="orange", label="AUC")
    ax.set_xlabel("Fold"); ax.set_ylabel("Score"); ax.set_title("TCAV CV"); ax.legend(); ax.grid(alpha=0.2, axis="y"); plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "tcav_cv_scores.png"), dpi=200)
    if show: plt.show()
    else: plt.close(fig)

    ref0 = refs[0]["idx"]
    cdew_vals = np.zeros(N)
    for i in tqdm(range(N), desc="C-DEW", unit="w"):
        j = ref0 if ref0 != i else max(0, i - 1)
        cdew_vals[i] = compute_cdew_distance(Ea_all[i], Ea_all[j], cam_all[i], cam_all[j], v_c, band=band,
                                              normalize_cam=normalize_cam, clip_cam_nonneg=clip_cam_nonneg)

    cdew_df = pd.DataFrame({"label_date": label_dates[:N], "is_event": is_event[:N],
                             "is_concept": concept_labels[:N], "cdew_ref": cdew_vals})
    cdew_df.to_csv(os.path.join(save_dir, "cdew_ref_fixed.csv"), index=False)

    # effects
    effect_rows = []
    ev1, ev0 = cdew_vals[is_event[:N] == 1], cdew_vals[is_event[:N] == 0]
    if len(ev1) > 2 and len(ev0) > 2:
        d, p, es = _paired_perm(ev1[:min(len(ev1), len(ev0))], ev0[:min(len(ev1), len(ev0))], n_perm)
        effect_rows.append(dict(effect_name="event_vs_nonevent", effect_size=es, p_raw=p, p_fdr=np.nan, n=min(len(ev1), len(ev0))))
    c1, c0 = cdew_vals[concept_labels[:N] == 1], cdew_vals[concept_labels[:N] == 0]
    if len(c1) > 2 and len(c0) > 2:
        d, p, es = _paired_perm(c1[:min(len(c1), len(c0))], c0[:min(len(c1), len(c0))], n_perm)
        effect_rows.append(dict(effect_name="concept_vs_nonconcept", effect_size=es, p_raw=p, p_fdr=np.nan, n=min(len(c1), len(c0))))
    ne_mask = is_event[:N] == 0
    c1_ne, c0_ne = cdew_vals[ne_mask & (concept_labels[:N] == 1)], cdew_vals[ne_mask & (concept_labels[:N] == 0)]
    if len(c1_ne) > 2 and len(c0_ne) > 2:
        d, p, es = _paired_perm(c1_ne[:min(len(c1_ne), len(c0_ne))], c0_ne[:min(len(c1_ne), len(c0_ne))], n_perm)
        effect_rows.append(dict(effect_name="concept_controlling_event", effect_size=es, p_raw=p, p_fdr=np.nan, n=min(len(c1_ne), len(c0_ne))))
    eff_df = pd.DataFrame(effect_rows)
    if fdr_correction and len(eff_df) > 0:
        eff_df["p_fdr"] = _benjamini_hochberg(eff_df["p_raw"].values)
    eff_df.to_csv(os.path.join(save_dir, "cdew_effects.csv"), index=False)

    # example pair detail
    i_ex, j_ex = min(10, N - 1), ref0
    if j_ex == i_ex: j_ex = max(0, i_ex - 1)
    _, detail = compute_cdew_distance(Ea_all[i_ex], Ea_all[j_ex], cam_all[i_ex], cam_all[j_ex], v_c,
                                       band=band, normalize_cam=normalize_cam, clip_cam_nonneg=clip_cam_nonneg, return_detail=True)
    pd.DataFrame({k: detail[k] for k in ("Ca", "Cb", "ga", "gb", "Wca", "Wcb")}).to_csv(
        os.path.join(save_dir, "example_pair_detail.csv"), index=False)

    # plots
    for data, labels, title, fname in [
        ([ev0, ev1], ["non-event", "event"], "C-DEW | event", "cdew_box_event.png"),
        ([c0, c1], ["non-concept", "concept"], "C-DEW | concept", "cdew_box_concept.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(data, labels=labels); ax.set_title(title); ax.grid(alpha=0.2, axis="y"); plt.tight_layout()
        fig.savefig(os.path.join(save_dir, fname), dpi=200)
        if show: plt.show()
        else: plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(cdew_df["label_date"], cdew_df["cdew_ref"], lw=1)
    ax.set_title("C-DEW over time"); ax.grid(alpha=0.2); plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "cdew_over_time.png"), dpi=200)
    if show: plt.show()
    else: plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(detail["Wca"], label="Wca"); ax.plot(detail["Wcb"], label="Wcb")
    ax.set_title("Concept-weighted (example)"); ax.legend(); ax.grid(alpha=0.2); plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "concept_weighted_example_pair.png"), dpi=200)
    if show: plt.show()
    else: plt.close(fig)

    return dict(cdew_df=cdew_df, effects_df=eff_df, cv_df=cv, v_c=v_c, tcav=tcav, concept_labels=concept_labels)


# ═══════════════════════════════════════════════════════════════════
# E: ConceptDash — multi-concept analysis
# ═══════════════════════════════════════════════════════════════════


def _create_concept_generic(df_raw, train_end, te_index, seq_len, N,
                            concept_name, pos_condition_fn, threshold_source="train_only"):
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
    vtrx_module=None,
    ew_module=None,
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
    _ensure(save_dir)
    np.random.seed(seed)
    train_end = str(meta["df_tr"].index[-1])
    te_index = meta["df_te"].index

    # labels
    label_cols: dict = {"label_date": label_dates[:N], "is_event": is_event[:N]}
    concept_names: List[str] = []
    for cname, cfn in concept_definitions.items():
        try:
            labels = _create_concept_generic(df_raw, train_end, te_index, cfg.seq_len, N, cname, cfn, threshold_source)
        except Exception as e:
            print(f"[ConceptDash] WARNING: {cname} failed: {e}")
            labels = np.zeros(N, dtype=np.int8)
        label_cols[cname] = labels
        concept_names.append(cname)
    concept_df = pd.DataFrame(label_cols)
    concept_df.to_csv(os.path.join(save_dir, "concept_labels.csv"), index=False)

    # prevalence
    pd.DataFrame([dict(concept=cn, positive_rate=float(concept_df[cn].mean()),
                        positive_count=int(concept_df[cn].sum()), threshold_source=threshold_source)
                   for cn in concept_names]).to_csv(os.path.join(save_dir, "concept_prevalence.csv"), index=False)

    # co-occurrence
    cooc_rows = []
    for i, ci in enumerate(concept_names):
        for j, cj in enumerate(concept_names):
            if j <= i: continue
            a, b = concept_df[ci].values, concept_df[cj].values
            inter, union_ = (a & b).sum(), (a | b).sum()
            jaccard = inter / union_ if union_ > 0 else 0.0
            n11, n00, n10, n01 = inter, ((1 - a) & (1 - b)).sum(), (a & (1 - b)).sum(), ((1 - a) & b).sum()
            denom = np.sqrt(float((n11 + n10) * (n11 + n01) * (n00 + n10) * (n00 + n01)) + 1e-12)
            cooc_rows.append(dict(concept_i=ci, concept_j=cj, jaccard=jaccard, phi_corr=(n11 * n00 - n10 * n01) / denom))
    cooc_df = pd.DataFrame(cooc_rows)
    cooc_df.to_csv(os.path.join(save_dir, "concept_cooccurrence.csv"), index=False)

    # per-concept TCAV
    tcav_rows = []
    concept_cavs: Dict[str, np.ndarray] = {}
    for cn in concept_names:
        y = concept_df[cn].values
        if y.sum() < 3 or (len(y) - y.sum()) < 3:
            tcav_rows.append(dict(concept=cn, cv_acc_mean=np.nan, cv_acc_std=np.nan, cv_auc_mean=np.nan, cv_auc_std=np.nan, pos_count=int(y.sum())))
            continue
        t = TCAVExtractorCV(cv_folds=tcav_cv_folds, seed=seed)
        t.fit(Ea_all[:N], y)
        cv = t.get_cv_df()
        tcav_rows.append(dict(concept=cn, cv_acc_mean=cv["accuracy"].mean(), cv_acc_std=cv["accuracy"].std(),
                              cv_auc_mean=cv["roc_auc"].mean(), cv_auc_std=cv["roc_auc"].std(), pos_count=int(y.sum())))
        concept_cavs[cn] = t.get_cav()
    pd.DataFrame(tcav_rows).to_csv(os.path.join(save_dir, "tcav_scores_by_concept.csv"), index=False)

    # per-concept C-DEW
    ref0 = refs[0]["idx"]
    cdew_cols: dict = {"label_date": label_dates[:N], "is_event": is_event[:N]}
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
    cdew_by_concept.to_csv(os.path.join(save_dir, "cdew_by_concept.csv"), index=False)

    # conditional event rates
    cond_rows = []
    for cn in concept_names:
        y_ev, y_cn = concept_df["is_event"].values, concept_df[cn].values
        on, off = y_cn == 1, y_cn == 0
        m1, l1, h1 = _bootstrap_ci(y_ev[on]) if on.sum() > 0 else (np.nan, np.nan, np.nan)
        m2, l2, h2 = _bootstrap_ci(y_ev[off]) if off.sum() > 0 else (np.nan, np.nan, np.nan)
        cond_rows.append(dict(concept=cn, pos=int(on.sum()),
                              p_event_given_concept=m1, p_event_given_concept_ci_low=l1, p_event_given_concept_ci_high=h1,
                              p_event_given_nonconcept=m2, p_event_given_nonconcept_ci_low=l2, p_event_given_nonconcept_ci_high=h2))
    pd.DataFrame(cond_rows).to_csv(os.path.join(save_dir, "conditional_event_rates.csv"), index=False)

    # top-k dates
    for cn in concept_names:
        col = f"cdew_{cn}"
        if col not in cdew_by_concept.columns: continue
        vals = cdew_by_concept[col].values
        if np.all(np.isnan(vals)): continue
        topk = np.argsort(-np.nan_to_num(vals))[:save_topk]
        pd.DataFrame(dict(rank=np.arange(1, len(topk) + 1), label_date=label_dates[topk],
                          cdew_value=vals[topk], is_event=is_event[topk],
                          concept_on=concept_df[cn].values[topk])).to_csv(
            os.path.join(save_dir, f"top10_dates_{cn}.csv"), index=False)

    # dashboard plot
    n_concepts = len(concept_names)
    fig, axes = plt.subplots(n_concepts + 2, 1, figsize=(14, 3 * (n_concepts + 2)), sharex=True)
    axes[0].fill_between(range(N), 0, is_event[:N], step="mid", alpha=0.3, color="red", label="event")
    axes[0].set_ylabel("Event"); axes[0].set_title("ConceptDash"); axes[0].legend(fontsize=8)
    for ci, cn in enumerate(concept_names):
        axes[ci + 1].fill_between(range(N), 0, concept_df[cn].values, step="mid", alpha=0.3, label=cn)
        axes[ci + 1].set_ylabel(cn[:15]); axes[ci + 1].legend(fontsize=8)
    for cn in concept_names:
        col = f"cdew_{cn}"
        if col in cdew_by_concept.columns:
            axes[-1].plot(cdew_by_concept[col].values, label=cn, alpha=0.7, lw=1)
    axes[-1].set_ylabel("C-DEW"); axes[-1].set_xlabel("Test index"); axes[-1].legend(fontsize=7); axes[-1].grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "dashboard_main.png"), dpi=150)
    if show: plt.show()
    else: plt.close(fig)

    # per-concept boxplots
    for cn in concept_names:
        col = f"cdew_{cn}"
        if col not in cdew_by_concept.columns: continue
        vals = cdew_by_concept[col].values
        v0 = vals[concept_df[cn].values == 0]; v1 = vals[concept_df[cn].values == 1]
        v0, v1 = v0[np.isfinite(v0)], v1[np.isfinite(v1)]
        if len(v0) > 0 and len(v1) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.boxplot([v0, v1], labels=[f"non-{cn}", cn])
            ax.set_title(f"C-DEW | {cn}"); ax.grid(alpha=0.2, axis="y"); plt.tight_layout()
            fig.savefig(os.path.join(save_dir, f"boxplot_{cn}.png"), dpi=200)
            if show: plt.show()
            else: plt.close(fig)

    # co-occurrence heatmap
    if len(cooc_df) > 0 and len(concept_names) > 1:
        mat = np.eye(len(concept_names))
        name2i = {n: i for i, n in enumerate(concept_names)}
        for _, row in cooc_df.iterrows():
            i, j = name2i.get(row["concept_i"]), name2i.get(row["concept_j"])
            if i is not None and j is not None:
                mat[i, j] = mat[j, i] = row["phi_corr"]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(concept_names))); ax.set_xticklabels(concept_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(concept_names))); ax.set_yticklabels(concept_names, fontsize=8)
        plt.colorbar(im, ax=ax, label="Phi"); ax.set_title("Co-occurrence"); plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "concept_cooccurrence_heatmap.png"), dpi=200)
        if show: plt.show()
        else: plt.close(fig)

    # C-DEW over time by concept
    fig, ax = plt.subplots(figsize=(14, 4))
    for cn in concept_names:
        col = f"cdew_{cn}"
        if col in cdew_by_concept.columns:
            ax.plot(label_dates[:N], cdew_by_concept[col].values, label=cn, alpha=0.7, lw=1)
    ax.set_title("C-DEW over time by concept"); ax.legend(fontsize=8); ax.grid(alpha=0.2); plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "cdew_over_time_by_concept.png"), dpi=200)
    if show: plt.show()
    else: plt.close(fig)

    return dict(concept_df=concept_df, cdew_by_concept=cdew_by_concept,
                concept_cavs=concept_cavs, cooc_df=cooc_df)
