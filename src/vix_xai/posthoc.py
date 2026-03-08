from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

DistMethod = Literal["wasserstein", "energy", "mmd", "quantile"]
EmbDistMethod = Literal["l2", "cosine", "energy", "mmd"]
WeightMode = Literal["local", "penalty"]


@dataclass(frozen=True)
class DTWResult:
    cost_matrix: np.ndarray
    acc_cost: np.ndarray
    path: List[Tuple[int, int]]
    mapping: np.ndarray
    total_cost: float
    normalized_cost: float
    local_cost_per_i: np.ndarray


def _as_1d(x: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size == 0:
        raise ValueError("empty sequence")
    if not np.all(np.isfinite(a)):
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def _as_2d(x: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected 2D array, got shape={a.shape}")
    if a.shape[0] == 0:
        raise ValueError("empty sequence")
    if not np.all(np.isfinite(a)):
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def _local_slice(T: int, t: int, k: int) -> slice:
    lo = max(0, t - k)
    hi = min(T, t + k + 1)
    return slice(lo, hi)


def _quantile_grid(n: int = 9) -> np.ndarray:
    return np.linspace(0.1, 0.9, int(n))


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return 0.0
    qs = _quantile_grid(19)
    qa = np.quantile(a, qs)
    qb = np.quantile(b, qs)
    return float(np.mean(np.abs(qa - qb)))


def quantile_l2_1d(a: np.ndarray, b: np.ndarray, n_q: int = 9) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return 0.0
    qs = _quantile_grid(n_q)
    qa = np.quantile(a, qs)
    qb = np.quantile(b, qs)
    return float(np.sqrt(np.mean((qa - qb) ** 2)))


def energy_distance_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return 0.0
    xa = a.reshape(-1, 1)
    xb = b.reshape(1, -1)
    exy = np.mean(np.abs(xa - xb))
    exx = np.mean(np.abs(xa - xa.T)) if a.size > 1 else 0.0
    eyy = np.mean(np.abs(b.reshape(-1, 1) - b.reshape(1, -1))) if b.size > 1 else 0.0
    d = 2.0 * exy - exx - eyy
    return float(max(d, 0.0))


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    X2 = np.sum(X * X, axis=1, keepdims=True)
    Y2 = np.sum(Y * Y, axis=1, keepdims=True).T
    D2 = X2 + Y2 - 2.0 * (X @ Y.T)
    return np.exp(-gamma * np.maximum(D2, 0.0))


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.size == 0 or Y.size == 0:
        return 0.0
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    if gamma is None:
        Z = np.vstack([X, Y])
        if Z.shape[0] <= 1:
            gamma = 1.0
        else:
            Z2 = np.sum(Z * Z, axis=1, keepdims=True)
            D2 = Z2 + Z2.T - 2.0 * (Z @ Z.T)
            tri = D2[np.triu_indices(D2.shape[0], k=1)]
            med = float(np.median(tri)) if tri.size > 0 else 1.0
            gamma = 1.0 / (med + 1e-12)

    Kxx = _rbf_kernel(X, X, gamma)
    Kyy = _rbf_kernel(Y, Y, gamma)
    Kxy = _rbf_kernel(X, Y, gamma)

    m = X.shape[0]
    n = Y.shape[0]

    term_xx = (np.sum(Kxx) - np.trace(Kxx)) / max(m * (m - 1), 1)
    term_yy = (np.sum(Kyy) - np.trace(Kyy)) / max(n * (n - 1), 1)
    term_xy = np.mean(Kxy)
    mmd2 = term_xx + term_yy - 2.0 * term_xy
    return float(max(mmd2, 0.0))


def _normalize_importance(g: Optional[np.ndarray], top_p: Optional[float] = None) -> Optional[np.ndarray]:
    if g is None:
        return None

    g = np.asarray(g, dtype=np.float64).reshape(-1)
    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    if g.size == 0:
        return None

    gmin, gmax = float(np.min(g)), float(np.max(g))
    if gmax > gmin:
        g = (g - gmin) / (gmax - gmin)
    else:
        g = np.zeros_like(g)

    if top_p is not None:
        tp = float(top_p)
        if not (0.0 < tp <= 1.0):
            raise ValueError("top_p must be in (0,1]")
        thr = np.quantile(g, 1.0 - tp)
        g = np.where(g >= thr, g, 0.0)
        gmax2 = float(np.max(g))
        if gmax2 > 0:
            g = g / gmax2
    return g


def compute_cost_matrix_dtdw_1d(
    a: Union[np.ndarray, Sequence[float]],
    b: Union[np.ndarray, Sequence[float]],
    k: int = 3,
    method: DistMethod = "wasserstein",
    band: Optional[int] = 5,
) -> np.ndarray:
    a = _as_1d(a)
    b = _as_1d(b)
    T = a.size
    if b.size != T:
        raise ValueError(f"DTDW expects equal length. got len(a)={T}, len(b)={b.size}")
    if k < 0:
        raise ValueError("k must be >= 0")
    if band is not None and band < 0:
        raise ValueError("band must be >= 0 or None")

    C = np.full((T, T), np.inf, dtype=np.float64)

    for i in range(T):
        js = range(T) if band is None else range(max(0, i - band), min(T, i + band + 1))
        ai = a[_local_slice(T, i, k)]
        for j in js:
            bj = b[_local_slice(T, j, k)]
            if method == "wasserstein":
                C[i, j] = wasserstein_1d(ai, bj)
            elif method == "energy":
                C[i, j] = energy_distance_1d(ai, bj)
            elif method == "mmd":
                C[i, j] = mmd_rbf(ai.reshape(-1, 1), bj.reshape(-1, 1))
            elif method == "quantile":
                C[i, j] = quantile_l2_1d(ai, bj, n_q=9)
            else:
                raise ValueError(f"unknown method: {method}")

    return C


def compute_cost_matrix_embedding(
    Ea: Union[np.ndarray, Sequence[Sequence[float]]],
    Eb: Union[np.ndarray, Sequence[Sequence[float]]],
    method: EmbDistMethod = "l2",
    k: int = 0,
    band: Optional[int] = 5,
) -> np.ndarray:
    A = _as_2d(Ea)
    B = _as_2d(Eb)
    if A.shape != B.shape:
        raise ValueError(f"Embedding sequences must have same shape. got {A.shape} vs {B.shape}")
    T, _ = A.shape
    if band is not None and band < 0:
        raise ValueError("band must be >= 0 or None")
    if k < 0:
        raise ValueError("k must be >= 0")

    C = np.full((T, T), np.inf, dtype=np.float64)

    if k == 0 and method in ("l2", "cosine"):
        if method == "l2":
            A2 = np.sum(A * A, axis=1, keepdims=True)
            B2 = np.sum(B * B, axis=1, keepdims=True).T
            D2 = A2 + B2 - 2.0 * (A @ B.T)
            D = np.sqrt(np.maximum(D2, 0.0))
            if band is None:
                return D
            for i in range(T):
                j0 = max(0, i - band)
                j1 = min(T, i + band + 1)
                C[i, j0:j1] = D[i, j0:j1]
            return C

        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        S = An @ Bn.T
        D = 1.0 - np.clip(S, -1.0, 1.0)
        if band is None:
            return D
        for i in range(T):
            j0 = max(0, i - band)
            j1 = min(T, i + band + 1)
            C[i, j0:j1] = D[i, j0:j1]
        return C

    for i in range(T):
        js = range(T) if band is None else range(max(0, i - band), min(T, i + band + 1))
        Ai = A[_local_slice(T, i, k)]
        for j in js:
            Bj = B[_local_slice(T, j, k)]
            if method == "energy":
                XY = np.sqrt(
                    np.maximum(
                        np.sum(Ai * Ai, axis=1, keepdims=True)
                        + np.sum(Bj * Bj, axis=1, keepdims=True).T
                        - 2.0 * (Ai @ Bj.T),
                        0.0,
                    )
                )
                exy = float(np.mean(XY))
                if Ai.shape[0] > 1:
                    XX = np.sqrt(
                        np.maximum(
                            np.sum(Ai * Ai, axis=1, keepdims=True)
                            + np.sum(Ai * Ai, axis=1, keepdims=True).T
                            - 2.0 * (Ai @ Ai.T),
                            0.0,
                        )
                    )
                    exx = float(np.mean(XX))
                else:
                    exx = 0.0
                if Bj.shape[0] > 1:
                    YY = np.sqrt(
                        np.maximum(
                            np.sum(Bj * Bj, axis=1, keepdims=True)
                            + np.sum(Bj * Bj, axis=1, keepdims=True).T
                            - 2.0 * (Bj @ Bj.T),
                            0.0,
                        )
                    )
                    eyy = float(np.mean(YY))
                else:
                    eyy = 0.0
                C[i, j] = max(2.0 * exy - exx - eyy, 0.0)
            elif method == "mmd":
                C[i, j] = mmd_rbf(Ai, Bj, gamma=None)
            else:
                raise ValueError("For k>0, method must be 'energy' or 'mmd'.")

    return C


def dtw_from_cost_matrix(C: np.ndarray, band: Optional[int] = 5, normalize: bool = True) -> DTWResult:
    C = np.asarray(C, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"cost matrix must be square. got {C.shape}")
    T = C.shape[0]

    D = np.full((T, T), np.inf, dtype=np.float64)
    D[0, 0] = C[0, 0]

    for i in range(T):
        j_range = range(T) if band is None else range(max(0, i - band), min(T, i + band + 1))
        for j in j_range:
            if i == 0 and j == 0:
                continue
            best_prev = np.inf
            if i > 0:
                best_prev = min(best_prev, D[i - 1, j])
            if j > 0:
                best_prev = min(best_prev, D[i, j - 1])
            if i > 0 and j > 0:
                best_prev = min(best_prev, D[i - 1, j - 1])
            D[i, j] = C[i, j] + best_prev

    total = float(D[T - 1, T - 1])

    i, j = T - 1, T - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        candidates = []
        if i > 0:
            candidates.append((D[i - 1, j], i - 1, j))
        if j > 0:
            candidates.append((D[i, j - 1], i, j - 1))
        if i > 0 and j > 0:
            candidates.append((D[i - 1, j - 1], i - 1, j - 1))
        _, ni, nj = min(candidates, key=lambda x: x[0])
        i, j = ni, nj
        path.append((i, j))
    path.reverse()

    mapping = np.full((T,), np.nan, dtype=np.float64)
    by_i: Dict[int, List[int]] = {}
    for ii, jj in path:
        by_i.setdefault(ii, []).append(jj)

    for ii in range(T):
        if ii in by_i:
            mapping[ii] = float(np.mean(by_i[ii]))

    last = None
    for ii in range(T):
        if np.isfinite(mapping[ii]):
            last = mapping[ii]
        elif last is not None:
            mapping[ii] = last

    last = None
    for ii in reversed(range(T)):
        if np.isfinite(mapping[ii]):
            last = mapping[ii]
        elif last is not None:
            mapping[ii] = last

    mapping = np.where(np.isfinite(mapping), np.rint(mapping), 0.0).astype(int)

    local = np.zeros((T,), dtype=np.float64)
    for ii in range(T):
        jj = int(mapping[ii])
        if 0 <= jj < T and np.isfinite(C[ii, jj]):
            local[ii] = float(C[ii, jj])

    norm_cost = total / max(len(path), 1) if normalize else total

    return DTWResult(
        cost_matrix=C,
        acc_cost=D,
        path=path,
        mapping=mapping,
        total_cost=total,
        normalized_cost=float(norm_cost),
        local_cost_per_i=local,
    )


def apply_event_weighting(
    C: np.ndarray,
    g_a: Optional[np.ndarray],
    g_b: Optional[np.ndarray],
    alpha: float = 1.0,
    top_p: Optional[float] = None,
    mode: WeightMode = "local",
    gamma: float = 1.0,
    penalty_fn: Optional[Callable[[int, int, int], float]] = None,
) -> np.ndarray:
    C = np.asarray(C, dtype=np.float64)
    T = C.shape[0]
    if C.shape != (T, T):
        raise ValueError("C must be square")

    ga = _normalize_importance(g_a, top_p=top_p)
    gb = _normalize_importance(g_b, top_p=top_p)

    if ga is None or gb is None:
        return C.copy()

    if ga.size != T or gb.size != T:
        raise ValueError(f"importance must match T={T}. got {ga.size}, {gb.size}")

    if penalty_fn is None:
        def penalty_fn(i: int, j: int, T_: int) -> float:
            d = abs(i - j) / max(T_ - 1, 1)
            return float(d * d)

    W = 1.0 + float(alpha) * (ga.reshape(-1, 1) + gb.reshape(1, -1)) / 2.0

    if mode == "local":
        return C * W

    if mode == "penalty":
        P = np.zeros_like(C)
        for i in range(T):
            for j in range(T):
                if np.isfinite(C[i, j]):
                    P[i, j] = float(penalty_fn(i, j, T))
                else:
                    P[i, j] = np.inf
        return C + float(gamma) * W * P

    raise ValueError(f"unknown mode: {mode}")


def dtdw_1d(
    a: Union[np.ndarray, Sequence[float]],
    b: Union[np.ndarray, Sequence[float]],
    k: int = 3,
    method: DistMethod = "wasserstein",
    band: Optional[int] = 5,
    normalize: bool = True,
) -> DTWResult:
    C = compute_cost_matrix_dtdw_1d(a, b, k=k, method=method, band=band)
    return dtw_from_cost_matrix(C, band=band, normalize=normalize)


def wdtdw_1d(
    a: Union[np.ndarray, Sequence[float]],
    b: Union[np.ndarray, Sequence[float]],
    g_a: Optional[np.ndarray],
    g_b: Optional[np.ndarray],
    k: int = 3,
    method: DistMethod = "wasserstein",
    band: Optional[int] = 5,
    normalize: bool = True,
    alpha: float = 1.0,
    top_p: Optional[float] = None,
    weight_mode: WeightMode = "local",
    gamma: float = 1.0,
) -> DTWResult:
    C0 = compute_cost_matrix_dtdw_1d(a, b, k=k, method=method, band=band)
    Cw = apply_event_weighting(C0, g_a=g_a, g_b=g_b, alpha=alpha, top_p=top_p, mode=weight_mode, gamma=gamma)
    return dtw_from_cost_matrix(Cw, band=band, normalize=normalize)


def dtdw_embedding(
    Ea: Union[np.ndarray, Sequence[Sequence[float]]],
    Eb: Union[np.ndarray, Sequence[Sequence[float]]],
    method: EmbDistMethod = "l2",
    k: int = 0,
    band: Optional[int] = 5,
    normalize: bool = True,
) -> DTWResult:
    C = compute_cost_matrix_embedding(Ea, Eb, method=method, k=k, band=band)
    return dtw_from_cost_matrix(C, band=band, normalize=normalize)


def wdtdw_embedding(
    Ea: Union[np.ndarray, Sequence[Sequence[float]]],
    Eb: Union[np.ndarray, Sequence[Sequence[float]]],
    g_a: Optional[np.ndarray],
    g_b: Optional[np.ndarray],
    emb_method: EmbDistMethod = "l2",
    emb_k: int = 0,
    band: Optional[int] = 5,
    normalize: bool = True,
    alpha: float = 1.0,
    top_p: Optional[float] = None,
    weight_mode: WeightMode = "local",
    gamma: float = 1.0,
) -> DTWResult:
    C0 = compute_cost_matrix_embedding(Ea, Eb, method=emb_method, k=emb_k, band=band)
    Cw = apply_event_weighting(C0, g_a=g_a, g_b=g_b, alpha=alpha, top_p=top_p, mode=weight_mode, gamma=gamma)
    return dtw_from_cost_matrix(Cw, band=band, normalize=normalize)
