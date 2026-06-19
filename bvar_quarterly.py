"""
BVAR (Bayesian Vector Autoregression) — Квартальная версия
Translated from R to Python
Requires: pandas, numpy, scipy, matplotlib, openpyxl, tqdm, statsmodels
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ================================================================
# ДАННЫЕ И ПРЕОБРАЗОВАНИЯ
# ================================================================

FILE_PATH = "/Users/scherbakovandrew/Documents/Данные для BVAR.xlsx"  

data = pd.read_excel(FILE_PATH)

cols = [
    "Date", "GPR", "US_GDP", "EU_GDP", "CN_GDP", "Urals",
    "GDP_(%)_q/q_real_2021", "unempl", "net_exp", "internal_debt",
    "budget", "usd_rub", "proces_ind", "Inflation_q/q_without_seas",
    "Interest_rate_(%)", "Fed_Bonds_3", "IMOEX", "M2X"
]
df = data[cols].copy().reset_index(drop=True)

# GDP + external GDPs: x/100 - 1
for col in ["GDP_(%)_q/q_real_2021", "EU_GDP", "US_GDP", "CN_GDP"]:
    vals = df[col].values.copy()
    for i in range(1, len(vals)):
        df.loc[i, col] = vals[i] / 100 - 1

# Inflation: x/100
vals = df["Inflation_q/q_without_seas"].values.copy()
for i in range(1, len(vals)):
    df.loc[i, "Inflation_q/q_without_seas"] = vals[i] / 100

# Log-diff
for col in ["IMOEX", "GPR", "Urals", "M2X", "internal_debt", "net_exp"]:
    vals = df[col].values.copy()
    for i in range(1, len(vals)):
        df.loc[i, col] = np.log(vals[i]) - np.log(vals[i - 1])

# x/100
for col in ["usd_rub", "proces_ind"]:
    vals = df[col].values.copy()
    for i in range(1, len(vals)):
        df.loc[i, col] = vals[i] / 100

# First diff of rates (/100)
for col in ["Interest_rate_(%)", "Fed_Bonds_3"]:
    vals = df[col].values.copy()
    for i in range(1, len(vals)):
        df.loc[i, col] = (vals[i] / 100) - (vals[i - 1] / 100)

# Drop first row
df = df.iloc[1:].reset_index(drop=True)
for col in df.columns:
    if col != "Date":
        df[col] = pd.to_numeric(df[col], errors="coerce")

dates     = pd.to_datetime(df["Date"])
y_data    = df.drop(columns=["Date"]).values.astype(float)
var_names = list(df.columns[1:])
N_vars    = len(var_names)

var_norm_names = {
    "GPR":                        "Geopolitics",
    "US_GDP":                     "US gdp",
    "EU_GDP":                     "EU gdp",
    "CN_GDP":                     "CN gdp",
    "Urals":                      "Oil",
    "GDP_(%)_q/q_real_2021":      "GDP",
    "unempl":                     "Unemployment",
    "net_exp":                    "Net Exp",
    "internal_debt":              "Debt",
    "budget":                     "Budget",
    "usd_rub":                    "Usd/rub",
    "proces_ind":                 "Proces_ind",
    "Inflation_q/q_without_seas": "Inflation",
    "Interest_rate_(%)":          "Int Rate",
    "Fed_Bonds_3":                "Bonds 3",
    "IMOEX":                      "IMOEX",
    "M2X":                        "M2X",
}

print("\n=== ADF Tests ===")
for col in var_names:
    series = df[col].dropna().values
    result = adfuller(series, autolag="AIC")
    print(f"  {col:<40s} ADF={result[0]:.4f}  p={result[1]:.4f}")

# ================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ================================================================

def create_var_lags(y: np.ndarray, p: int):
    T, N = y.shape
    Y = y[p:, :]
    X = np.zeros((T - p, N * p))
    for i in range(1, p + 1):
        X[:, (i - 1) * N: i * N] = y[p - i: T - i, :]
    X = np.column_stack([X, np.ones(T - p)])
    return Y, X


def setup_minnesota_prior(y: np.ndarray, p: int, lam: float = 0.2,
                           alpha: float = 2, var_scale: float = 1):
    T, N = y.shape
    K = N * p + 1
    phi_ar1 = np.zeros(N)
    sigma   = np.zeros(N)

    for i in range(N):
        yi = y[:, i][~np.isnan(y[:, i])]
        if len(yi) <= 2:
            phi_ar1[i] = 0.9
            sigma[i]   = np.std(yi, ddof=1) if len(yi) > 1 else 1e-4
        else:
            y_lag  = yi[:-1]
            y_lead = yi[1:]
            X_ar   = np.column_stack([np.ones(len(y_lag)), y_lag])
            beta_ar = np.linalg.lstsq(X_ar, y_lead, rcond=None)[0]
            phi_ar1[i] = beta_ar[1]
            resid = y_lead - X_ar @ beta_ar
            sigma[i] = np.sqrt(np.sum(resid ** 2) / max(len(y_lead) - 2, 1))

    B_prior = np.zeros((K, N))
    for i in range(N):
        B_prior[i, i] = phi_ar1[i]

    V_prior_list = []
    for eq in range(N):
        V_eq = np.zeros((K, K))
        for j in range(N):
            for l in range(1, p + 1):
                idx = (l - 1) * N + j
                if eq == j:
                    V_eq[idx, idx] = (lam ** 2) / (l ** (2 * alpha))
                else:
                    s_ratio = (sigma[eq] ** 2) / (sigma[j] ** 2) if sigma[j] != 0 else 1.0
                    V_eq[idx, idx] = (lam ** 2) * s_ratio / (l ** (2 * alpha))
        V_eq[K - 1, K - 1] = 100.0
        V_prior_list.append(V_eq)

    return {
        "B_prior":      B_prior,
        "V_prior_list": V_prior_list,
        "S_prior":      np.diag(sigma ** 2) * var_scale,
        "nu_prior":     N + 2,
        "sigma":        sigma,
        "phi_ar1":      phi_ar1,
    }


def _riwish(nu: int, Psi: np.ndarray) -> np.ndarray:
    """Sample from Inverse-Wishart(nu, Psi)."""
    n = Psi.shape[0]
    Psi_inv = np.linalg.inv(Psi + np.eye(n) * 1e-10)
    try:
        L = np.linalg.cholesky(Psi_inv + np.eye(n) * 1e-10)
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.abs(np.diag(Psi_inv))))
    Z = np.zeros((n, n))
    for i in range(n):
        Z[i, i] = np.sqrt(np.random.chisquare(max(nu - i, 1)))
        if i > 0:
            Z[:i, i] = np.random.randn(i)
    A = L @ Z
    W = A @ A.T
    return np.linalg.inv(W + np.eye(n) * 1e-10)


def gibbs_bvar(Y: np.ndarray, X: np.ndarray, prior: dict,
               n_draws: int = 5000, n_burn: int = 1000):
    T_obs, N = Y.shape
    K = X.shape[1]

    B_draws     = np.zeros((K, N, n_draws))
    Sigma_draws = np.zeros((N, N, n_draws))

    XtX = X.T @ X
    B   = np.linalg.lstsq(XtX + np.eye(K) * 0.01, X.T @ Y, rcond=None)[0]
    resid0 = Y - X @ B
    Sigma  = np.diag(np.diag(np.cov(resid0.T, ddof=1)))

    V_prior_inv_list = [
        np.linalg.inv(V + np.eye(K) * 1e-6)
        for V in prior["V_prior_list"]
    ]

    print("Running Gibbs sampler...")
    for it in tqdm(range(n_draws + n_burn)):
        for j in range(N):
            V_pri_inv_j = V_prior_inv_list[j]
            sig_jj = max(Sigma[j, j], 1e-10)
            V_post_inv = V_pri_inv_j + (1.0 / sig_jj) * XtX
            V_post = np.linalg.inv(V_post_inv + np.eye(K) * 1e-8)
            V_post = (V_post + V_post.T) / 2
            b_post = V_post @ (V_pri_inv_j @ prior["B_prior"][:, j] +
                                (1.0 / sig_jj) * X.T @ Y[:, j])
            try:
                L = np.linalg.cholesky(V_post + np.eye(K) * 1e-10)
                B[:, j] = b_post + L @ np.random.randn(K)
            except np.linalg.LinAlgError:
                B[:, j] = b_post

        resid   = Y - X @ B
        S_post  = prior["S_prior"] + resid.T @ resid
        nu_post = prior["nu_prior"] + T_obs
        Sigma   = _riwish(nu_post, S_post)

        if it >= n_burn:
            d = it - n_burn
            B_draws[:, :, d]     = B
            Sigma_draws[:, :, d] = Sigma

    return {"B_draws": B_draws, "Sigma_draws": Sigma_draws}

# ================================================================
# ОЦЕНКА BVAR НА ПОЛНОЙ ВЫБОРКЕ
# ================================================================

n_lags  = 1      # квартальные данные — 1 лаг
lam     = 0.4
alpha   = 2.5
n_draws = 5000
n_burn  = 1000
gdp_idx = var_names.index("GDP_(%)_q/q_real_2021")

Y, X     = create_var_lags(y_data, n_lags)
prior    = setup_minnesota_prior(y_data, n_lags, lam, alpha)
bvar_res = gibbs_bvar(Y, X, prior, n_draws, n_burn)

B_mean     = bvar_res["B_draws"].mean(axis=2)
Sigma_mean = bvar_res["Sigma_draws"].mean(axis=2)

# ================================================================
# ДИАГНОСТИКА 1: Geweke + ESS
# ================================================================

def geweke_z(chain: np.ndarray, first: float = 0.1, last: float = 0.5) -> float:
    n  = len(chain)
    a  = chain[:int(n * first)]
    b  = chain[n - int(n * last):]
    se_a = np.std(a, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 1e-10
    se_b = np.std(b, ddof=1) / np.sqrt(len(b)) if len(b) > 1 else 1e-10
    denom = np.sqrt(se_a ** 2 + se_b ** 2)
    return (np.mean(a) - np.mean(b)) / denom if denom > 1e-15 else 0.0


def effective_size(chain: np.ndarray, max_lag: int = 400) -> float:
    n = len(chain)
    c = chain - chain.mean()
    var = np.dot(c, c)
    if var == 0:
        return float(n)
    lags = min(max_lag, n // 2)
    acf_vals = np.array([np.dot(c[:n - k], c[k:]) / var for k in range(1, lags)])
    cutoff = np.where(np.abs(acf_vals) < 2 / np.sqrt(n))[0]
    tau = 1 + 2 * np.sum(acf_vals[:cutoff[0]]) if len(cutoff) > 0 else 1 + 2 * np.sum(np.abs(acf_vals))
    return n / max(tau, 1.0)


def sig_stars(z: float) -> str:
    az = abs(z)
    return "***" if az >= 2.576 else (" **" if az >= 1.960 else ("  *" if az >= 1.645 else "   "))


print("\n=== Convergence Diagnostics ===")
print(f"{'Параметр':<40s}  {'Geweke z':>10s}  {'ESS':>10s}")
print("-" * 63)

all_z = []

print("\n  [B] Собственные лаги (диагональ):")
for i in range(N_vars):
    chain = bvar_res["B_draws"][i, i, :]
    gz    = geweke_z(chain)
    ess   = effective_size(chain)
    all_z.append(gz)
    print(f"  B[{i},{i}] ({var_names[i]:<28s})  {gz:8.3f} {sig_stars(gz)}  {ess:8.0f}")

print("\n  [Σ] Дисперсии уравнений (диагональ):")
for i in range(N_vars):
    chain = bvar_res["Sigma_draws"][i, i, :]
    gz    = geweke_z(chain)
    ess   = effective_size(chain)
    all_z.append(gz)
    print(f"  Σ[{i},{i}] ({var_names[i]:<28s})  {gz:8.3f} {sig_stars(gz)}  {ess:8.0f}")

print(f"\nСошлось (|z|<2): {sum(abs(z) < 2 for z in all_z)} из {len(all_z)} "
      f"({100 * np.mean([abs(z) < 2 for z in all_z]):.1f}%)  |  Макс. |z|: {max(abs(z) for z in all_z):.3f}")

# Trace plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
chain_b = bvar_res["B_draws"][gdp_idx, gdp_idx, :]
chain_s = bvar_res["Sigma_draws"][gdp_idx, gdp_idx, :]

axes[0, 0].plot(chain_b, color="blue", lw=0.8)
axes[0, 0].axhline(chain_b.mean(), color="red", lw=2)
axes[0, 0].set_title("Trace: GDP own lag-1"); axes[0, 0].set_xlabel("Draw")

axes[0, 1].plot(chain_s, color="darkgreen", lw=0.8)
axes[0, 1].axhline(chain_s.mean(), color="red", lw=2)
axes[0, 1].set_title("Trace: Sigma[GDP,GDP]"); axes[0, 1].set_xlabel("Draw")

n_acf = 40
acf_v = np.array([np.corrcoef(chain_b[:-k], chain_b[k:])[0, 1] for k in range(1, n_acf + 1)])
axes[1, 0].bar(range(1, n_acf + 1), acf_v)
axes[1, 0].axhline(0, color="black")
axes[1, 0].set_title("ACF: GDP own lag-1")

axes[1, 1].hist(chain_b, bins=50, color="lightblue", edgecolor="gray")
axes[1, 1].axvline(prior["phi_ar1"][gdp_idx], color="red", lw=2, linestyle="--", label="Prior AR(1)")
axes[1, 1].set_title("Posterior: GDP own lag-1"); axes[1, 1].legend()

plt.tight_layout()
plt.savefig("trace_plots_quarterly.pdf")
plt.show()
print("Saved: trace_plots_quarterly.pdf")

# ================================================================
# ДИАГНОСТИКА 2: In-sample fit + Ljung-Box
# ================================================================

print("\n=== In-Sample Fit ===")
Y_hat         = X @ B_mean
residuals_mat = Y - Y_hat
for i, name in enumerate(var_names):
    ss_res = np.sum(residuals_mat[:, i] ** 2)
    ss_tot = np.sum((Y[:, i] - Y[:, i].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    print(f"  {name:<40s} R² = {r2:.4f}")

print("\n=== Ljung-Box Test (lag=12) ===")
for i, name in enumerate(var_names):
    lb   = acorr_ljungbox(residuals_mat[:, i], lags=[12], return_df=True)
    stat = lb["lb_stat"].values[0]
    pval = lb["lb_pvalue"].values[0]
    flag = "(!)" if pval < 0.05 else ""
    print(f"  {name:<40s} stat={stat:.2f}  p={pval:.4f}  {flag}")

# ================================================================
# IRF
# ================================================================

def compute_irf_bvar(B_draws: np.ndarray, Sigma_draws: np.ndarray,
                     n_periods: int = 12, conf_level: float = 0.68):
    K, N, n_dr = B_draws.shape
    p = (K - 1) // N
    irf_draws = np.zeros((N, N, n_periods, n_dr))

    print("\nComputing IRFs...")
    for d in tqdm(range(n_dr)):
        B_d     = B_draws[:, :, d]
        Sigma_d = Sigma_draws[:, :, d]
        try:
            P = np.linalg.cholesky(Sigma_d + np.eye(N) * 1e-10)
        except np.linalg.LinAlgError:
            P = np.diag(np.sqrt(np.abs(np.diag(Sigma_d))))

        A = B_d[:N * p, :].T   # N x (N*p)
        if p > 1:
            comp = np.zeros((N * p, N * p))
            comp[:N, :] = A
            comp[N:, :N * (p - 1)] = np.eye(N * (p - 1))
        else:
            comp = A  # N x N when p=1

        pwr = np.eye(N)
        for h in range(n_periods):
            if h > 0:
                pwr = pwr @ comp
            irf_draws[:, :, h, d] = pwr[:N, :N] @ P

    lo_q = (1 - conf_level) / 2
    return {
        "mean":  irf_draws.mean(axis=3),
        "lower": np.quantile(irf_draws, lo_q, axis=3),
        "upper": np.quantile(irf_draws, 1 - lo_q, axis=3),
        "draws": irf_draws,
    }


n_periods   = 12
irf_results = compute_irf_bvar(bvar_res["B_draws"], bvar_res["Sigma_draws"], n_periods)


def plot_var_irf(irf_results: dict, var_names: list, var_norm_names: dict,
                 target_idx: int, n_periods: int = 12, save_prefix: str = "irf"):
    target_name = var_norm_names.get(var_names[target_idx], var_names[target_idx])
    shock_indices = [i for i in range(len(var_names)) if i != target_idx]
    n_shocks = len(shock_indices)
    ncols = 4
    nrows = int(np.ceil(n_shocks / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes = axes.flatten()
    periods = np.arange(n_periods)

    for ax_i, shock_idx in enumerate(shock_indices):
        shock_name = var_norm_names.get(var_names[shock_idx], var_names[shock_idx])
        mean  = irf_results["mean"][target_idx, shock_idx, :]
        lower = irf_results["lower"][target_idx, shock_idx, :]
        upper = irf_results["upper"][target_idx, shock_idx, :]
        ax = axes[ax_i]
        ax.fill_between(periods, lower, upper, alpha=0.3, color="lightblue")
        ax.plot(periods, mean, color="blue", lw=1.5)
        ax.axhline(0, color="black", alpha=0.4)
        ax.set_title(f"{target_name} → {shock_name}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Periods"); ax.set_ylabel("Response")

    for ax in axes[n_shocks:]:
        ax.set_visible(False)

    fig.suptitle(f"IRFs: {target_name} Response", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = f"{save_prefix}_{target_name.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=120)
    plt.show()
    print(f"Saved: {fname}")


inf_idx = var_names.index("Inflation_q/q_without_seas")
m2x_idx = var_names.index("M2X")

print("\n=== IRFs ===")
plot_var_irf(irf_results, var_names, var_norm_names, gdp_idx, save_prefix="irf_q")
plot_var_irf(irf_results, var_names, var_norm_names, inf_idx, save_prefix="irf_q")
plot_var_irf(irf_results, var_names, var_norm_names, m2x_idx, save_prefix="irf_q")

# ================================================================
# FEVD
# ================================================================

def compute_fevd(irf_results: dict, n_periods: int = 12) -> np.ndarray:
    irf_mean = irf_results["mean"]
    N = irf_mean.shape[0]
    fevd = np.zeros((N, N, n_periods))
    for h in range(1, n_periods + 1):
        for resp in range(N):
            total_var = np.sum(irf_mean[resp, :, :h] ** 2)
            for shock in range(N):
                fevd[resp, shock, h - 1] = (
                    np.sum(irf_mean[resp, shock, :h] ** 2) / total_var * 100
                    if total_var > 0 else 0.0
                )
    return fevd


fevd_results = compute_fevd(irf_results, n_periods)
print(f"\n=== FEVD for {var_norm_names[var_names[gdp_idx]]} ===")
for h in [1, 3, 6, 12]:
    print(f"  Horizon {h}:")
    for shock in range(len(var_names)):
        nm = var_norm_names.get(var_names[shock], var_names[shock])
        print(f"    {nm:<20s}: {fevd_results[gdp_idx, shock, h - 1]:6.2f}%")

# ================================================================
# POSTERIOR PREDICTIVE CHECK
# ================================================================

print("\n=== Posterior Predictive Check ===")
n_check  = min(1000, n_draws)
draw_idx = np.random.choice(n_draws, size=n_check, replace=False)
N_v = len(var_names)

Y_sim = np.zeros((Y.shape[0], Y.shape[1], n_check))
for d_i, d in enumerate(draw_idx):
    mean_mat = X @ bvar_res["B_draws"][:, :, d]
    S = bvar_res["Sigma_draws"][:, :, d]
    try:
        L = np.linalg.cholesky(S + np.eye(N_v) * 1e-10)
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.abs(np.diag(S))))
    Y_sim[:, :, d_i] = mean_mat + np.random.randn(Y.shape[0], N_v) @ L.T

print(f"{'Variable':<20s} {'Act.Mean':>10s} {'Sim.Mean':>10s} {'Act.SD':>10s} {'Sim.SD':>10s} {'p-value':>10s}")
print("-" * 82)
for i, name in enumerate(var_names):
    nm        = var_norm_names.get(name, name)
    act_mean  = Y[:, i].mean()
    sim_means = Y_sim[:, i, :].mean(axis=0)
    tt        = stats.ttest_1samp(sim_means, popmean=act_mean)
    sig       = "***" if tt.pvalue < 0.01 else (" **" if tt.pvalue < 0.05 else
                ("  *" if tt.pvalue < 0.10 else "   "))
    print(f"{nm:<20s} {act_mean:>10.5f} {sim_means.mean():>10.5f} "
          f"{Y[:, i].std():>10.5f} {Y_sim[:, i, :].std(axis=0).mean():>10.5f} "
          f"{tt.pvalue:>10.4f} {sig}")

# ================================================================
# ROLLING BVAR OOS FORECAST
# ================================================================

def forecast_bvar_multistep(bvar_train: dict, train_y: np.ndarray,
                             p: int, forecast_horizon: int,
                             target_col_idx: int) -> np.ndarray:
    N  = train_y.shape[1]
    nd = bvar_train["B_draws"].shape[2]
    te = train_y.shape[0]
    fc_draws = np.zeros((forecast_horizon, nd))

    for d in range(nd):
        B_draw = bvar_train["B_draws"][:, :, d]
        y_lag  = train_y[te - 1: te - p - 1: -1, :].copy()
        state  = y_lag.flatten()

        for h in range(forecast_horizon):
            x_now  = np.append(state, 1.0)
            y_next = x_now @ B_draw
            fc_draws[h, d] = y_next[target_col_idx]
            if p > 1:
                state = np.concatenate([y_next, state[:N * (p - 1)]])
            else:
                state = y_next

    return fc_draws


horizons   = [1, 2, 3]
start_date = pd.Timestamp("2021-01-01")
start_idx  = next((i for i, d in enumerate(dates) if d >= start_date), None)
min_obs    = n_lags + 20

print("\n### Rolling BVAR GDP forecast (OOS evaluation) ###")
bvar_oos = {}

for fh in horizons:
    print(f"\n  Horizon h={fh}")
    errors = []
    rows   = []

    for t in tqdm(range(start_idx, len(y_data) - fh)):
        if t < min_obs:
            continue
        train_y = y_data[:t, :]
        Y_t, X_t = create_var_lags(train_y, n_lags)
        if Y_t.shape[0] < n_lags + 10:
            continue
        prior_t  = setup_minnesota_prior(train_y, n_lags, lam, alpha)
        bvar_t   = gibbs_bvar(Y_t, X_t, prior_t, n_draws=2000, n_burn=500)
        fc_draws = forecast_bvar_multistep(bvar_t, train_y, n_lags, fh, gdp_idx)
        fc_h     = fc_draws[fh - 1, :]
        actual_val = y_data[t + fh, gdp_idx]
        if np.isnan(actual_val):
            continue
        errors.append(actual_val - fc_h.mean())
        rows.append({
            "date":     dates.iloc[t + fh],
            "actual":   actual_val,
            "forecast": fc_h.mean(),
            "lower80":  np.quantile(fc_h, 0.10),
            "upper80":  np.quantile(fc_h, 0.90),
        })

    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mae  = np.mean(np.abs(errors))
    ft   = pd.DataFrame(rows)
    bvar_oos[f"h{fh}"] = {"table": ft, "errors": errors, "rmse": rmse, "mae": mae}
    print(f"  RMSE={rmse:.6f}  MAE={mae:.6f}  N={len(errors)}")

# Rolling OOS plots
fig, axes = plt.subplots(len(horizons), 1, figsize=(12, 4 * len(horizons)))
for ax_i, fh in enumerate(horizons):
    res = bvar_oos[f"h{fh}"]
    ft  = res["table"]
    ax  = axes[ax_i] if len(horizons) > 1 else axes
    ax.fill_between(ft["date"], ft["lower80"], ft["upper80"], alpha=0.4, color="lightblue")
    ax.plot(ft["date"], ft["actual"],   color="black", lw=1.2, label="Фактическое")
    ax.plot(ft["date"], ft["forecast"], color="red",   lw=1.2, linestyle="--", label="Прогноз BVAR")
    ax.set_title(f"BVAR GDP: горизонт h={fh} (RMSE={res['rmse']:.5f})", fontweight="bold")
    ax.set_xlabel("Дата"); ax.set_ylabel("ВВП q/q")
    ax.legend(loc="lower left")
fig.suptitle("BVAR: Rolling OOS прогноз ВВП (квартальный)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("bvar_rolling_oos_quarterly.png", dpi=120)
plt.show()
print("Saved: bvar_rolling_oos_quarterly.png")

# ================================================================
# ПРОГНОЗ НА 1 И 2 КВАРТАЛА ВПЕРЁД (fan chart)
# ================================================================

print("\n" + "=" * 65)
print("   ПРОГНОЗ ВВП НА 1 И 2 КВАРТАЛА ВПЕРЁД (вся выборка)")
print("=" * 65)

horizons_future = [1, 2]
te_full = y_data.shape[0]
N_full  = y_data.shape[1]
max_fh  = max(horizons_future)

all_paths = np.zeros((max_fh, n_draws))
for d in range(n_draws):
    B_draw = bvar_res["B_draws"][:, :, d]
    y_lag  = y_data[te_full - 1: te_full - n_lags - 1: -1, :].copy()
    state  = y_lag.flatten()
    for h in range(max_fh):
        x_now  = np.append(state, 1.0)
        y_next = x_now @ B_draw
        all_paths[h, d] = y_next[gdp_idx]
        if n_lags > 1:
            state = np.concatenate([y_next, state[:N_full * (n_lags - 1)]])
        else:
            state = y_next

last_date    = dates.iloc[-1]
# Quarterly offset
future_dates = [last_date + pd.DateOffset(months=3 * i) for i in range(1, max_fh + 1)]

for fh in horizons_future:
    fc    = all_paths[fh - 1, :]
    fdate = future_dates[fh - 1]
    print(f"\n--- Прогноз на {fh} кв. вперёд (дата: {fdate.strftime('%Y-%m')}) ---")
    print(f"  Медиана:           {np.median(fc)*100:+.4f}%")
    print(f"  Среднее:           {np.mean(fc)*100:+.4f}%")
    print(f"  80% ДИ:  [{np.quantile(fc, 0.10)*100:+.4f}% ; {np.quantile(fc, 0.90)*100:+.4f}%]")
    print(f"  90% ДИ:  [{np.quantile(fc, 0.05)*100:+.4f}% ; {np.quantile(fc, 0.95)*100:+.4f}%]")

fan_df = pd.DataFrame({
    "date":  future_dates,
    "med":   np.quantile(all_paths, 0.50, axis=1),
    "mean":  all_paths.mean(axis=1),
    "lo90":  np.quantile(all_paths, 0.05, axis=1),
    "hi90":  np.quantile(all_paths, 0.95, axis=1),
    "lo80":  np.quantile(all_paths, 0.10, axis=1),
    "hi80":  np.quantile(all_paths, 0.90, axis=1),
    "lo50":  np.quantile(all_paths, 0.25, axis=1),
    "hi50":  np.quantile(all_paths, 0.75, axis=1),
})

n_hist  = 20   # ~5 лет квартальных
hist_s  = slice(max(0, te_full - n_hist), te_full)
hist_df = pd.DataFrame({
    "date": dates.iloc[hist_s].values,
    "gdp":  y_data[hist_s, gdp_idx],
})

fig, ax = plt.subplots(figsize=(12, 6))
ax.fill_between(fan_df["date"], fan_df["lo90"] * 100, fan_df["hi90"] * 100,
                color="#2171b5", alpha=0.15, label="90% ДИ")
ax.fill_between(fan_df["date"], fan_df["lo80"] * 100, fan_df["hi80"] * 100,
                color="#2171b5", alpha=0.20, label="80% ДИ")
ax.fill_between(fan_df["date"], fan_df["lo50"] * 100, fan_df["hi50"] * 100,
                color="#2171b5", alpha=0.25, label="50% ДИ")
ax.plot(hist_df["date"], hist_df["gdp"] * 100, color="black", lw=1.5, label="Фактический ВВП")
ax.plot(fan_df["date"],  fan_df["med"] * 100,  color="blue", lw=1.5,
        linestyle="--", label="Медиана прогноза")
for fh in horizons_future:
    ax.scatter([future_dates[fh - 1]], [fan_df["med"].iloc[fh - 1] * 100],
               color="red", zorder=5, s=60)
ax.axvline(last_date, color="grey", linestyle=":", alpha=0.8)
ax.set_title("BVAR: Прогноз ВВП на 1 и 2 квартала вперёд", fontsize=13, fontweight="bold")
ax.set_xlabel(f"Дата  (последнее наблюдение: {last_date.strftime('%Y-%m')})")
ax.set_ylabel("ВВП q/q, %")
ax.legend(loc="lower left")
fig.text(0.5, -0.01, "Заливка: 50% / 80% / 90% доверительные интервалы",
         ha="center", fontsize=9, color="gray")
plt.tight_layout()
plt.savefig("bvar_fan_chart_quarterly.png", dpi=120)
plt.show()
print("Saved: bvar_fan_chart_quarterly.png")

# Итоговая таблица
print(f"\n--- Полная таблица прогноза (h = 1 ... {max_fh}) ---")
print(f"  {'Дата':<12s} {'Медиана%':>10s} {'Среднее%':>10s} {'80% ДИ нижн.':>14s} {'80% ДИ верхн.':>14s}")
print("-" * 67)
for i in range(max_fh):
    print(f"  {future_dates[i].strftime('%Y-%m'):<12s} "
          f"{fan_df['med'].iloc[i]*100:>10.3f} "
          f"{fan_df['mean'].iloc[i]*100:>10.3f} "
          f"{fan_df['lo80'].iloc[i]*100:>14.3f} "
          f"{fan_df['hi80'].iloc[i]*100:>14.3f}")
