import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Any, Optional
from scipy.linalg import cho_factor, solve_triangular
from scipy.linalg import LinAlgError
from scipy.special import gammaln
from scipy.stats import norm
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import arviz as az
import pymc as pm
from scipy.stats import norm
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from scipy.signal import detrend
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tools.eval_measures import rmse, aic
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import solve
from statsmodels.tsa.filters.hp_filter import hpfilter
from openpyxl import load_workbook
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.preprocessing import StandardScaler
import warnings
#from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from numpy.linalg import LinAlgError
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import gammaln

warnings.filterwarnings("ignore")
excel_file = r'/Users/scherbakovandrew/Documents/Модель_new.xlsx'

@dataclass
class VARResult:
    Y: np.ndarray
    X: np.ndarray  
    Y0: np.ndarray
    prior: Dict[str, Any]
    logdensy: float
    lags: int
    beta: np.ndarray
    sigma: np.ndarray
    beta_draws: Optional[np.ndarray] = None
    sigma_draws: Optional[np.ndarray] = None

def _varlags(y: np.ndarray, P: int):
    """Create VAR lag matrix"""
    T, N = y.shape
    Y = y[P:, :]
    X = np.empty((T - P, P * N))
    
    for p in range(P):
        X[:, p*N:(p+1)*N] = y[P-1-p:T-1-p, :]
    
    return Y, X

def _logdet(A: np.ndarray) -> float:
    """Compute log determinant safely"""
    try:
        sign, logdet = np.linalg.slogdet(A)
        return logdet if sign > 0 else -np.inf
    except:
        return -np.inf

def _multgammaln(N: int, a: float) -> float:
    """Multivariate log gamma function"""
    result = 0.25 * N * (N - 1) * np.log(np.pi)
    for j in range(N):
        result += gammaln(a - 0.5 * j)
    return result

def var_dummyobsprior(y: np.ndarray,
                      w: np.ndarray,
                      n_draws: int,
                      info: Dict[str, Any],
                      verbosity: int = 1) -> VARResult:
    """Python port of VAR_dummyobsprior.m

    y: (T, N) endogenous data
    w: (T, k_exog) exogenous data
    info: dict with keys 'lags', and optional 'minnesota', 'simsdummy', 'trspl', 'dummyobs'
    """
    Tfull, N = y.shape
    P = int(info.get("lags", 1))
    if P <= 0:
        raise ValueError("info.lags must be >=1")

    T = Tfull - P
    k_exog = 0 if w is None else w.shape[1]
    K = P * N + k_exog

    if verbosity:
        print("VAR_dummyobsprior")
        print(f"lags: {P}")

    # Priors
    if "dummyobs" in info:
        Yprior = np.asarray(info["dummyobs"].get("Y"))
        Xprior = np.asarray(info["dummyobs"].get("X"))
        vprior = float(info["dummyobs"].get("v", 0.0))
    else:
        Yprior = np.zeros((0, N))
        Xprior = np.zeros((0, K))
        vprior = 0.0

        # Minnesota prior
        mn = info.get("minnesota", {})
        if "tightness" in mn and not np.isinf(mn["tightness"]):
            exog_std = float(mn.get("exog_std", 1e8))
            decay = float(mn.get("decay", 1))

            # оценка стандартных отклонений остатков AR(4)
            if "sigma" not in mn:
                sigma_data = np.asarray(mn.get("sigma_data", y))
                sigma = np.zeros(N)
                sigma_arlags = int(mn.get("sigma_arlags", max(0, min(P, sigma_data.shape[0] - 3))))
                for n in range(N):
                    if sigma_arlags == 0:
                        resid = sigma_data[:, n] - np.mean(sigma_data[:, n])
                    else:
                        yn, ylagsn = _varlags(sigma_data[:, [n]], sigma_arlags)
                        Xn = np.hstack([ylagsn, np.ones((yn.shape[0], 1))])
                        bn, *_ = np.linalg.lstsq(Xn, yn, rcond=None)
                        resid = yn.flatten() - (Xn @ bn).flatten()
                    sigma[n] = np.std(resid, ddof=1)
                mn["sigma"] = sigma
            else:
                sigma = np.asarray(mn["sigma"]).astype(float)

            if "sigma_factor" in mn:
                sigma = sigma * float(mn["sigma_factor"])

            # Prior for coefficients
            Winv = (np.arange(1, P + 1, dtype=float) ** decay)
            Winv = np.kron(Winv, 1/(sigma / float(mn["tightness"])))
            Winv = np.concatenate([Winv, (exog_std ** -1) * np.ones(k_exog)])
            Winv_mat = np.diag(Winv)

            B0 = np.zeros((K, N))
            mvector = np.asarray(mn.get("mvector", np.ones(N)))
            mvector = mvector[:N]
            # задание значений коэффициентов при первых собственных лагах
            for i in range(N):
                B0[i, i] = mvector[i]

            Yprior = np.vstack([Yprior, Winv_mat @ B0])
            Xprior = np.vstack([Xprior, Winv_mat])

            # Prior for variance
            sigma_deg = float(mn.get("sigma_deg", N + 2))
            Z = np.diag(sigma * np.sqrt(max(sigma_deg - N - 1, 1e-12)))
            Yprior = np.vstack([Yprior, Z])
            Xprior = np.vstack([Xprior, np.zeros((N, K))])
            vprior += sigma_deg

            if verbosity:
                print("Minnesota prior")
                print(f"Note: for a proper prior need sigma_deg > {N - 1}")
                print(f"Note: for E(Sigma) to exist need sigma_deg > {N + 1}")

    # Полные матрицы данных
    Y, X = _varlags(y, P)
    if k_exog > 0:
        X = np.hstack([X, w[P:, :]])
    Y0 = y[:P, :]
    print(f'приоры: {Xprior}, {Yprior}')
    #Xprior = np.array([])
    #Yprior = np.array([])
    # Объединение
    Yst = np.vstack([Yprior, Y]) if Yprior.size else Y
    Xst = np.vstack([Xprior, X]) if Xprior.size else X
    print(Xst, Yst)
    # Marginal likelihood
    if Xprior.size:
        logdetXtXprior = _logdet(Xprior.T @ Xprior)
        Bprior, *_ = np.linalg.lstsq(Xprior, Yprior, rcond=None)
        Uprior = Yprior - Xprior @ Bprior
        Sprior = Uprior.T @ Uprior
        logdetSprior = _logdet(Sprior)
    else:
        logdetXtXprior = 0.0
        Sprior = np.eye(N)
        logdetSprior = 0.0

    logdetXtXst = _logdet(Xst.T @ Xst)
    Bst, *_ = np.linalg.lstsq(Xst, Yst, rcond=None)
    Ust = Yst - Xst @ Bst
    Sst = Ust.T @ Ust
    logdetSst = _logdet(Sst)

    if verbosity:
        print(f"Prior degrees of freedom of Sigma: {vprior}")
        if vprior <= (N - 1):
            print("Prior for Sigma is improper!")

    logdensy = (-0.5 * N * T * np.log(np.pi)
                + 0.5 * N * (logdetXtXprior - logdetXtXst)
                + _multgammaln(N, 0.5 * (T + vprior)) - _multgammaln(N, 0.5 * vprior)
                + 0.5 * vprior * logdetSprior - 0.5 * (T + vprior) * logdetSst)
    print(f'logdetSst: {logdetSst}')
    # Posterior moments
    beta = Bst 
    sigma = Sst / max(T + vprior - N - 1, 1.0)

    beta_draws = None
    sigma_draws = None
    
    # генерация МСМС-выборок
    if n_draws and n_draws > 0:
        nf = int(round(T + vprior))
        XtX = Xst.T @ Xst
        cF, lower = cho_factor(XtX)
        chol_inv = solve_triangular(cF, np.eye(cF.shape[0]), lower=lower)
        XtXinv_chol = chol_inv.T

        try:
            Sst_chol = np.linalg.cholesky(Sst)
        except LinAlgError:
            Sst_chol = np.linalg.cholesky(Sst + 1e-10 * np.eye(N))

        beta_draws = np.empty((K, N, n_draws))
        sigma_draws = np.empty((N, N, n_draws))
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        
        for d in range(n_draws):
            Z = rng.standard_normal(size=(nf, N))
            IW = np.linalg.inv(Z.T @ Z)
            Sigma_draw = Sst_chol @ IW @ Sst_chol.T
            E = rng.standard_normal(size=(K, N))
            B_draw = Bst + XtXinv_chol @ E @ np.linalg.cholesky(Sigma_draw)
            beta_draws[:, :, d] = B_draw
            sigma_draws[:, :, d] = Sigma_draw

    return VARResult(
        Y=Y,
        X=X,
        Y0=Y0,
        prior={},
        logdensy=float(logdensy),
        lags=P,
        beta=beta,
        sigma=sigma,
        beta_draws=beta_draws,
        sigma_draws=sigma_draws,
    )


def print_shock_sizes(var_result, var_names, y_data, shock_size=1.0, var_norm_names=None):
   
    if var_norm_names is None:
        var_norm_names = {name: name for name in var_names}
    
    sigma = var_result.sigma
    try:
        chol_sigma = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        chol_sigma = np.linalg.cholesky(sigma + 1e-10 * np.eye(sigma.shape[0]))
    
    # Стандартные отклонения остатков
    resid_std = np.sqrt(np.diag(sigma))
    
    # Диагональ структурного шока
    structural_shock_std = np.diag(chol_sigma)
    
    # Фактический размер шока
    applied_shock = shock_size * structural_shock_std
    
    # Среднее по абсолютным значениям
    mean_abs = np.nanmean(np.abs(y_data), axis=0)
    
    # Относительный размер шока (относительно среднего)
    relative_shock = np.divide(
        applied_shock, 
        mean_abs, 
        out=np.full_like(applied_shock, np.nan), 
        where=mean_abs != 0
    )
    
    display_names = [var_norm_names.get(name, name) for name in var_names]
    
    shock_table = pd.DataFrame({
        'Variable': display_names,
        'Mean(|x|) (Data)': mean_abs,
        'Residual Std (Reduced Form)': resid_std,
        'Structural Shock Std (Cholesky)': structural_shock_std,
        f'Applied Shock Size (×{shock_size})': applied_shock,
        'Shock / Mean(|x|)': relative_shock
    })
    
   
    print(shock_table.round(6).to_string(index=False))

    
    return shock_table
def compute_impulse_responses(var_result: VARResult, n_periods: int = 20, shock_size: float = 1.0, confidence_level: float = 0.90):
    P = var_result.lags
    N = var_result.sigma.shape[0]
    n_draws = var_result.beta_draws.shape[2] if var_result.beta_draws is not None else 0
    print(P, N, n_draws)
    
    # Массив для хранения всех IRF из MCMC выборок
    if n_draws > 0:
        irf_draws = np.zeros((N, N, n_periods, n_draws))
        
        for draw in range(n_draws):
            # Получение коэффициентов для данной выборки
            beta_draw = var_result.beta_draws[:, :, draw]
            sigma_draw = var_result.sigma_draws[:, :, draw]
            
            A = beta_draw[:P*N, :].T
            A = A.reshape(N, P, N)
            
            # разложение Холецкого для данной выборки
            try:
                chol_sigma = np.linalg.cholesky(sigma_draw)
            except:
                chol_sigma = np.linalg.cholesky(sigma_draw + 1e-10 * np.eye(N))
            
            # VAR(1) 
            if P > 1:
                F = np.zeros((N*P, N*P))
                F[:N, :] = A.reshape(N, P*N)
                F[N:, :N*(P-1)] = np.eye(N*(P-1))
            else:
                F = A.reshape(N, N)
            
            if P > 1:
                G = np.zeros((N*P, N))
                G[:N, :] = chol_sigma
            else:
                G = chol_sigma
            
            # Вычисление IRF для данной выборки
            if P > 1:
                Phi = np.eye(N*P)
            else:
                Phi = np.eye(N)
            
            for h in range(n_periods):
                if P > 1:
                    irf_draws[:, :, h, draw] = (Phi @ G)[:N, :] * shock_size
                    Phi = Phi @ F
                else:
                    irf_draws[:, :, h, draw] = Phi @ G * shock_size
                    Phi = Phi @ F
        
        # Вычисление выборочных средних и стандартных отклонений
        irf_mean = np.mean(irf_draws, axis=3)
        irf_std = np.std(irf_draws, axis=3, ddof=1) 
        
        # Построение классических доверительных интервалов 
        alpha = 1 - confidence_level
        z_critical = norm.ppf(1 - alpha/2)  
        se_mean = irf_std / np.sqrt(n_draws)
        
        # Доверительные интервалы
        irf_lower = irf_mean - z_critical * se_mean
        irf_upper = irf_mean + z_critical * se_mean
              
        print(f"Используется {confidence_level*100:.1f}% доверительный интервал")
        print(f"Критическое значение (z): {z_critical:.3f}")
        
        return {
            'mean': irf_mean,
            'std': irf_std,
            'se_mean': se_mean,
            'lower': irf_lower,
            'upper': irf_upper,
            'draws': irf_draws,
            'confidence_level': confidence_level,
            'method': 'classical_normal'
        }
    
    else:
        # Если нет MCMC выборок, возвращаем только точечную оценку
        beta = var_result.beta
        A = beta[:P*N, :].T
        A = A.reshape(N, P, N)
        
        try:
            chol_sigma = np.linalg.cholesky(var_result.sigma)
        except:
            chol_sigma = np.linalg.cholesky(var_result.sigma + 1e-10 * np.eye(N))
        
        if P > 1:
            F = np.zeros((N*P, N*P))
            F[:N, :] = A.reshape(N, P*N)
            F[N:, :N*(P-1)] = np.eye(N*(P-1))
        else:
            F = A.reshape(N, N)
        
        if P > 1:
            G = np.zeros((N*P, N))
            G[:N, :] = chol_sigma
        else:
            G = chol_sigma
        
        irf = np.zeros((N, N, n_periods))
        
        if P > 1:
            Phi = np.eye(N*P)
        else:
            Phi = np.eye(N)
        
        for h in range(n_periods):
            if P > 1:
                irf[:, :, h] = (Phi @ G)[:N, :] * shock_size
                Phi = Phi @ F
            else:
                irf[:, :, h] = Phi @ G * shock_size
                Phi = Phi @ F
        
        return {'mean': irf, 'lower': None, 'upper': None, 'method': 'point_estimate'}

var_norm_names = {'GDP_(%)_m/m_real_2021': 'GDP', 
             'Interest_rate_(%)': 'Int Rate', 
             'Inflation_m/m_without_seas': 'Inflation', 
             'Fed_Bonds_10': 'Bonds 10', 
             'real_eff_exchange_rate_index_m/m': 'Exch Rate',
             'Brent': 'Oil',
             'IMOEX': 'IMOEX',
             #'consumption_real_2021': 'Consump',
             'M2X': 'M2X',
             'exp_inf_firms_seas': 'Exp inf firm',
             'spread_diff': 'Spread',
             'bud_balance': 'Deficit',
             'gov_expan': 'Expances',
             'unempl': 'Unemployment',
             'net_exp': 'Net export',
             'GPR': 'Geopolitics',
             'nom_eff_exch_rate_index_m/m': 'Nom Exch Rate'}



def plot_gdp_impulse_responses(irf_results, var_names, var_idx=0, n_periods=20):
    n_vars = len(var_names)
    n_shocks = n_vars - 1  # Исключаем собственный шок переменной
    fig, axes = plt.subplots(4, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    # Проверяем, есть ли доверительные интервалы
    has_ci = irf_results['lower'] is not None and irf_results['upper'] is not None
    method = irf_results.get('method', 'unknown')
    confidence_level = irf_results.get('confidence_level', 0.90)
    
    shock_idx = 0
    for i in range(n_vars):
        if i == var_idx:  # Пропускаем шок к самому себе
            continue
            
        ax = axes[shock_idx]
        
        # Импульсный отклик на шок переменной i
        if 'mean' in irf_results:
            response_mean = irf_results['mean'][var_idx, i, :n_periods]
        else:
            response_mean = irf_results['median'][var_idx, i, :n_periods]  
        
        # Вычисление кумулятивного эффекта
        irf_final = 1
        for irfss in response_mean:
            irf_final = irf_final * (1 + irfss)
        print(f'{var_names[i]}, {irf_final - 1:.6f}')
        
        periods = np.arange(n_periods)
        
        # Основная линия отклика
        label_main = f'IRF (Mean)' if method == 'classical_normal' else 'IRF'
        ax.plot(periods, response_mean, 'b-', linewidth=2.5, label=label_main)
        
        # Доверительные интервалы 
        if has_ci:
            response_lower = irf_results['lower'][var_idx, i, :n_periods]
            response_upper = irf_results['upper'][var_idx, i, :n_periods]
            
            ci_label = f'{confidence_level*100:.0f}% CI (Normal)' if method == 'classical_normal' else f'{confidence_level*100:.0f}% CI'
            ax.fill_between(periods, response_lower, response_upper, 
                          alpha=0.3, color='lightblue', label=ci_label)
            
            # Границы доверительного интервала
            ax.plot(periods, response_lower, '--', color='navy', alpha=0.7, linewidth=1)
            ax.plot(periods, response_upper, '--', color='navy', alpha=0.7, linewidth=1)
            
            # Добавляем информацию о стандартной ошибке если доступна
            if 'se_mean' in irf_results:
                se_mean = irf_results['se_mean'][var_idx, i, :n_periods]
                print(f"Стандартная ошибка для {var_names[i]} в период 1: {se_mean[0]:.6f}")
        
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.4, linewidth=0.8)
        
        # Настройка графика
        ax.set_title(f'{var_norm_names.get(var_names[var_idx])} response to {var_norm_names.get(var_names[i])} shock', fontsize=11, fontweight='bold')
        ax.set_xlabel('Periods', fontsize=10)
        ax.set_ylabel('Response', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Легенда только для первого графика
        if shock_idx == 0:
            ax.legend(fontsize=9, loc='upper right')
        
        # Улучшенное форматирование осей
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        # Установка лимитов для лучшего отображения
        y_max = np.max(np.abs(response_mean))
        if has_ci:
            ci_max = max(np.max(np.abs(response_lower)), np.max(np.abs(response_upper)))
            y_max = max(y_max, ci_max)
        
        if y_max > 0:
            ax.set_ylim(-y_max * 1.1, y_max * 1.1)
        
        shock_idx += 1
    
    # Удаляем лишние subplots
    for i in range(shock_idx, len(axes)):
        fig.delaxes(axes[i])
    
    ci_method_text = "Classical Normal Distribution" if method == 'classical_normal' else "Percentile-based"
    plt.suptitle(f'Impulse Response Functions: {var_norm_names.get(var_names[var_idx])} Response to Structural Shocks\n(with {confidence_level*100:.0f}% Confidence Intervals using {ci_method_text})', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def compute_cumulative_irf(irf_results):
    cum_irf = {}
    
    for key in irf_results:
        if key not in ['mean', 'median', 'lower', 'upper', 'std', 'se_mean', 'draws']:
            cum_irf[key] = irf_results[key]
    
    # средний отклик
    if 'mean' in irf_results:
        mean_irf = irf_results['mean']
        N, _, n_periods = mean_irf.shape
        cum_mean = np.zeros_like(mean_irf)
        
        for i in range(N):
            for j in range(N):
                cumulative_product = 1.0
                for h in range(n_periods):
                    cumulative_product *= (1.0 + mean_irf[i, j, h])
                    cum_mean[i, j, h] = cumulative_product - 1.0
        cum_irf['mean'] = cum_mean

    # Если есть MCMC-выборки, пересчитываем всё на их основе
    if 'draws' in irf_results and irf_results['draws'] is not None:
        draws = irf_results['draws']  # (N, N, n_periods, n_draws)
        N, _, n_periods, n_draws = draws.shape
        cum_draws = np.zeros_like(draws)
        
        # Рассчитываем накопленный эффект для каждой MCMC-выборки
        for d in range(n_draws):
            for i in range(N):
                for j in range(N):
                    cumulative_product = 1.0
                    for h in range(n_periods):
                        cumulative_product *= (1.0 + draws[i, j, h, d])
                        cum_draws[i, j, h, d] = cumulative_product - 1.0
        
        cum_irf['draws'] = cum_draws
        cum_irf['mean'] = cum_mean    #np.mean(cum_draws, axis=3)
        cum_irf['std'] = np.std(cum_draws, axis=3, ddof=1)
        cum_irf['se_mean'] = cum_irf['std'] / np.sqrt(n_draws)
        
        # Пересчитываем доверительные интервалы
        if 'confidence_level' in irf_results:
            alpha = 1 - irf_results['confidence_level']
            z_critical = norm.ppf(1 - alpha/2)
            cum_irf['lower'] = cum_irf['mean'] - z_critical * cum_irf['se_mean']
            cum_irf['upper'] = cum_irf['mean'] + z_critical * cum_irf['se_mean']
    
    return cum_irf


def plot_cumulative_impulse_responses(irf_results, var_names, var_idx=0, n_periods=20):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm

    n_vars = len(var_names)
    fig, axes = plt.subplots(4, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    has_ci = irf_results['lower'] is not None and irf_results['upper'] is not None
    method = irf_results.get('method', 'unknown')
    confidence_level = irf_results.get('confidence_level', 0.90)
    
    shock_idx = 0
    for i in range(n_vars):
        if i == var_idx:
            continue
            
        ax = axes[shock_idx]
        response_mean = irf_results['mean'][var_idx, i, :n_periods]
        periods = np.arange(n_periods)
        
        label_main = 'Cumulative IRF (Mean)' if method == 'classical_normal' else 'Cumulative IRF'
        ax.plot(periods, response_mean, 'b-', linewidth=2.5, label=label_main)
        
        # Доверительные интервалы
        if has_ci:
            response_lower = irf_results['lower'][var_idx, i, :n_periods]
            response_upper = irf_results['upper'][var_idx, i, :n_periods]
            ci_label = f'{confidence_level*100:.0f}% CI (Normal)' if method == 'classical_normal' else f'{confidence_level*100:.0f}% CI'
            ax.fill_between(periods, response_lower, response_upper, alpha=0.3, color='lightblue', label=ci_label)
            ax.plot(periods, response_lower, '--', color='navy', alpha=0.7, linewidth=1)
            ax.plot(periods, response_upper, '--', color='navy', alpha=0.7, linewidth=1)
        
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.4, linewidth=0.8)
        ax.set_title(f'Cumulative {var_norm_names.get(var_names[var_idx])} response to {var_norm_names.get(var_names[i])} shock', 
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Periods', fontsize=10)
        ax.set_ylabel('Cumulative Response', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if shock_idx == 0:
            ax.legend(fontsize=9, loc='upper right')
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        # Настройка осей
        y_max = np.max(np.abs(response_mean))
        if has_ci:
            ci_max = max(np.max(np.abs(response_lower)), np.max(np.abs(response_upper)))
            y_max = max(y_max, ci_max)
        if y_max > 0:
            ax.set_ylim(-y_max * 1.1, y_max * 1.1)
        
        shock_idx += 1
    
    for i in range(shock_idx, len(axes)):
        fig.delaxes(axes[i])
    
    ci_method_text = "Classical Normal Distribution" if method == 'classical_normal' else "Percentile-based"
    plt.suptitle(f'Cumulative Impulse Response Functions: {var_norm_names.get(var_names[var_idx])} Response\n(with {confidence_level*100:.0f}% CIs using {ci_method_text})', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def print_cumulative_irf_table(cumulative_irf, var_names, target_indices=[2, 6], n_periods=None):
    import pandas as pd
    import numpy as np
    
    # Определяем последний период
    if n_periods is None:
        n_periods = cumulative_irf['mean'].shape[2]
    last_period_idx = n_periods - 1

    shock_names = [var_norm_names.get(name, name) for name in var_names]
    table_data = {}
    
    for target_idx in target_indices:
        target_name = var_norm_names.get(var_names[target_idx], var_names[target_idx])
        responses = cumulative_irf['mean'][target_idx, :, last_period_idx]
        table_data[target_name] = responses 
    
    df_table = pd.DataFrame(table_data, index=shock_names)
    
    # Форматируем и выводим таблицу
    print(f"\nКумулятивные импульсные отклики за период {last_period_idx + 1} (в %):")
    print("=" * 70)
    print(df_table.round(4).to_string())
    print("=" * 70)
#Загрузка данных

data = pd.read_excel('/Users/scherbakovandrew/Documents/Model_gretl.xlsx')
df = pd.DataFrame(data)
df = pd.DataFrame({'Date': df['Date'], 'GPR': df['GPR'], 'Brent': df['Brent'], 'GDP_(%)_m/m_real_2021': df['GDP_(%)_m/m_real_2021'], 
                   'unempl': df['unempl'],
                   'net_exp': df['net_exp'], 'Inflation_m/m_without_seas': df['Inflation_m/m_without_seas'], 
                   'bud_balance': df['bud_balance'], 'gov_expan': df['gov_expan'], 'M2X': df['M2X'],
                   'Interest_rate_(%)': df['Interest_rate_(%)'], 'Fed_Bonds_10': df['Fed_Bonds_10'], 'nom_eff_exch_rate_index_m/m': df['nom_eff_exch_rate_index_m/m'],
                   'real_eff_exchange_rate_index_m/m': df['real_eff_exchange_rate_index_m/m'],
                   'spread_diff': df['spread_diff'], 'IMOEX': df['IMOEX'],
                   'exp_inf_firms_seas': df['exp_inf_firms_seas']})
for col in ['GDP_(%)_m/m_real_2021']:
    #df[col] = df[col] - seasonal_decompose(df[col], model='additive', period=12).seasonal
    for i in range(0, len(df[col])):
        df[col][i] = df[col][i]/100 - 1   
    
for col in ['nom_eff_exch_rate_index_m/m']:
    #df[col] = df[col] - seasonal_decompose(df[col], model='additive', period=12).seasonal
    for i in range(0, len(df[col])):
        df[col][i] = df[col][i]/100  
   
for col in ['IMOEX', 'GPR']:
    #df[col] = df[col] - seasonal_decompose(df[col], model='multiplicative', period=12).seasonal
    a = list(df[col])
    for i in range(1, len(df[col])):
        df[col][i] = a[i]/a[i-1] - 1   
        
##for col in ['volume_of_shares_mil']:
    ##df[col] = df[col] - seasonal_decompose(df[col], model='multiplicative', period=12).seasonal
    ##a = list(df[col])
    ##for i in range(1, len(df[col])):
        ##df[col][i] = a[i]/a[i-1] 

for col in ['Interest_rate_(%)', 'exp_inf_firms_seas']:
    a = list(df[col])
    for i in range(1, len(df[col])):
        df[col][i] = (a[i]/100)/(a[i-1]/100) - 1
        
#for col in ['consumption_real_2021']:
    ##df[col] = df[col] - seasonal_decompose(df[col], model='additive', period=12).seasonal
    #a = list(df[col])
    #for i in range(1, len(df[col])):
        #df[col][i] = np.log(a[i]) - np.log(a[i-1]) 
    

for col in ['Brent', 'M2X']:
    #df[col] = df[col] - seasonal_decompose(df[col], model='multiplicative', period=12).seasonal
    a = list(df[col])
    for i in range(1, len(df[col])):
        df[col][i] = a[i]/a[i-1] - 1
        #df[col][i] = np.log(df[col][i])
        
for col in ['Fed_Bonds_10', 'net_exp']:
    #df[col] = df[col] - seasonal_decompose(df[col], model='multiplicative', period=12).seasonal
    a = list(df[col])
    for i in range(1, len(df[col])):
        df[col][i] = (a[i]/100)/(a[i-1]/100) - 1


##for col in ['M2']:
    ##df[col] = seasonal_decompose(df[col], model='multiplicative', period=12).trend
    ##a = list(df[col])
    ##print(a)
    ##for i in range(1, len(df[col])):
        ##df[col][i] = a[i]/a[i-1]
for col in ['Inflation_m/m_without_seas']:
    a = list(df[col])
    for i in range(0, len(df[col])):
        df[col][i] = df[col][i]/100

df = df.drop(index=0)
##df = df.drop(index = range(8))
##df = df.drop(index = range(115, 119))
df = df.reset_index(drop = True)

#for col in ['consumption_real_2021']:
    #df[col] = detrend(df[col])

#  ДФ тест
def adf_test(series, title=''):
    result = adfuller(series.dropna())
    print(f'ADF Test for {title}')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    print()

for col in ['GDP_(%)_m/m_real_2021', 'Interest_rate_(%)', 'Inflation_m/m_without_seas', 'Fed_Bonds_10', 'real_eff_exchange_rate_index_m/m', 'Brent', 'IMOEX',
            'M2X', 'exp_inf_firms_seas', 'spread_diff', 'bud_balance', 'gov_expan', 'unempl', 'net_exp', 'nom_eff_exch_rate_index_m/m', 'GPR']:
    adf_test(df[col], title=col)

df['Date'] = pd.to_datetime(df['Date'])

df.set_index('Date', inplace=True)

df = df.apply(pd.to_numeric, errors='coerce')
var_names = df.columns.tolist()
with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name = 'Data', index=False)
y_full = df.values
print(df)

scaler = StandardScaler()
y_scaled = scaler.fit_transform(y_full)

print("Данные загружены:")
print(f"Размерность: {y_full.shape}")
print(f"Переменные: {var_names}")

# Настройка модели BVAR
info = {
    'lags': 8,  # Количество лагов
    'minnesota': {
        'tightness': 0.7,      # Степень сжатия Minnesota prior
        'sigma_deg': len(var_names) + 5,  # Степени свободы для приора на сигма
        'decay': 0.7, # Скорость убывания весов лагов
        'sigma_arlags': 1,  # GPR         Brent     GDP   consumption   unemp     exp    inflation    budget     expances     M2X     int rate    bonds   nom_exch  real_exch   spread      IMOEX     exp_inf         
        #'mvector': np.array([-0.228694, 0.316132, 0.117119, 0.975756, 0.593753, 0.991262, 0.297908, -0.00592306, -0.288041, 0.0164542, 0.211113, 0.148931, 0.305189, 0.197227, 0.0620511, -0.0911293, 0.159483]),
        #'sigma_factor': 0.1
    }
}

# Оценка BVAR модели
print("\nОценка BVAR модели...")
var_result = var_dummyobsprior(y_full, None, n_draws=5000, info=info, verbosity=1)

print(f"\nРезультаты оценки:")
print(f"Логарифмическая маргинальная плотность: {var_result.logdensy:.2f}")
print(f"Количество лагов: {var_result.lags}")
print(f"Размерность beta: {var_result.beta.shape}")
print(f"Размерность sigma: {var_result.sigma.shape}")

print("\nВычисление импульсных откликов...")    

# Размер шоков
shock_table = print_shock_sizes(
    var_result, 
    var_names, 
    y_data=y_full,             
    shock_size=1.0, 
    var_norm_names=var_norm_names
)

# Вычисление импульсных откликов
print("\nВычисление импульсных откликов...")    
irf_results = compute_impulse_responses(var_result, n_periods=8, shock_size= 1.0)
cumulative_irf = compute_cumulative_irf(irf_results)

if 'draws' in irf_results and irf_results['draws'] is not None:
    print(f"Размерность IRF draws: {irf_results['draws'].shape}")
    print("Доверительные интервалы будут построены на основе MCMC выборок")
else:
    print("Доверительные интервалы недоступны (нет MCMC выборок)")

# Построение графиков импульсных откликов ВВП
print("\nПостроение графиков импульсных откликов ВВП с доверительными интервалами...")
plot_gdp_impulse_responses(irf_results, var_names, var_idx=2, n_periods=8)
plot_gdp_impulse_responses(irf_results, var_names, var_idx=5, n_periods=8)
plot_cumulative_impulse_responses(cumulative_irf, var_names, var_idx=2, n_periods=8)
plot_cumulative_impulse_responses(cumulative_irf, var_names, var_idx=5, n_periods=8)
#plot_gdp_impulse_responses(irf_results, var_names, var_idx=9, n_periods=6)


# Вывод некоторых числовых результатов

has_ci = irf_results['lower'] is not None and irf_results['upper'] is not None
print_cumulative_irf_table(cumulative_irf, var_names, target_indices=[2, 5], n_periods=8)
#for i, var_name in enumerate(var_names):
    #if i == 0:  # Пропускаем сам ВВП
        #continue
    #print(f"\nШок {var_name}:")
    #for t in range(5):
        #median_val = irf_results['median'][0, i, t]
        #if has_ci:
            #lower_val = irf_results['lower'][0, i, t]
            #upper_val = irf_results['upper'][0, i, t]
            #print(f"  Период {t+1}: {median_val:.6f} [{lower_val:.6f}, {upper_val:.6f}]")
        #else:
            #print(f"  Период {t+1}: {median_val:.6f}")
            


