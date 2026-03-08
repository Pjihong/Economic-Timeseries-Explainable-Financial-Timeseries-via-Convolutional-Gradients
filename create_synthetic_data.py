"""Generate synthetic timeseries_data.csv for testing."""
import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 2000

dates = pd.bdate_range("2016-01-04", periods=N)

def gbm(s0, mu, sigma, n):
    dt = 1 / 252
    ret = np.random.normal(mu * dt, sigma * np.sqrt(dt), n)
    return s0 * np.cumprod(1 + ret)

def mean_revert(s0, mu, kappa, sigma, n):
    vals = [s0]
    for _ in range(n - 1):
        v = vals[-1] + kappa * (mu - vals[-1]) + sigma * np.random.randn()
        vals.append(max(v, 5))
    return np.array(vals)

data = {
    "날짜": dates,
    "SPX": gbm(2100, 0.08, 0.15, N),
    "VIX": mean_revert(15, 18, 0.05, 1.5, N),
    "KOSPI": gbm(2000, 0.05, 0.18, N),
    "Gold": gbm(1200, 0.03, 0.12, N),
    "WTI": gbm(45, 0.02, 0.30, N),
    "DXY": mean_revert(96, 97, 0.02, 0.5, N),
    "Silver": gbm(16, 0.02, 0.20, N),
    "Copper": gbm(2.5, 0.03, 0.18, N),
    "USD/GBP": mean_revert(0.75, 0.76, 0.01, 0.005, N),
    "USD/CNY": mean_revert(6.8, 6.9, 0.005, 0.02, N),
    "USD/JPY": mean_revert(110, 108, 0.01, 0.5, N),
    "USD/EUR": mean_revert(0.88, 0.87, 0.01, 0.005, N),
    "USD/CAD": mean_revert(1.30, 1.32, 0.01, 0.005, N),
}

# Inject VIX spike (simulate crisis)
data["VIX"][990:1020] += np.random.uniform(10, 25, 30)
data["SPX"][990:1020] *= np.random.uniform(0.95, 0.99, 30)

df = pd.DataFrame(data)
out_path = os.path.join(os.path.dirname(__file__), "timeseries_data.csv")
df.to_csv(out_path, index=False)
print(f"Created {out_path}: shape={df.shape}")
print(f"Columns: {list(df.columns)}")
