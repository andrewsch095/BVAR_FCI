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

#Задаем класс для хранения результатов модели
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
    beta_draws: Optional[np.ndarray] = None #выборка наборов оценок параметров после МК
    sigma_draws: Optional[np.ndarray] = None    

def _varlags(y: np.ndarray, lags: int):
    #Создаёт лаговые матрицы для VAR.
    T = y.shape[0]
    if lags <= 0 or lags >= T:
        raise ValueError("Некорректное число лагов.")
    Y = y[lags:, :]
    X = np.hstack([y[lags - p - 1:-p - 1, :] for p in range(lags)])
    return Y, X

def _logdet(A: np.ndarray) -> float:
    #Вычисляет log|A| с обработкой знака
    sign, logdet_val = np.linalg.slogdet(A)
    if sign <= 0:
        raise ValueError("Матрица не положительно определена для logdet.")
    return logdet_val

def _multgammaln(a: float, p: int) -> float:
    #Многомерная гамма-функция.
    from scipy.special import gammaln
    p_int = int(p)  # гарантируем целое
    return (p_int * (p_int - 1) * np.log(np.pi) / 4.0 +
            np.sum([gammaln(a + (1.0 - j) / 2.0) for j in range(1, p_int + 1)]))


warnings.filterwarnings("ignore")
excel_file = r'/Users/scherbakovandrew/Documents/Модель_new.xlsx'
df = pd.read_excel('/Users/scherbakovandrew/Documents/Model_gretl.xlsx')

#Преобразование рядов для стационарности

for col in ['GDP_(%)_m/m_real_2021']:
    #df[col] = df[col] - seasonal_decompose(df[col], model='additive', period=12).seasonal
    for i in range(0, len(df[col])):
        df[col][i] = df[col][i]/100 - 1    
   
for col in ['IMOEX']:
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
        
for col in ['consumption_real_2021']:
    #df[col] = df[col] - seasonal_decompose(df[col], model='additive', period=12).seasonal
    a = list(df[col])
    for i in range(1, len(df[col])):
        df[col][i] = np.log(a[i]) - np.log(a[i-1]) 
    

for col in ['Brent', 'M2']:
    #df[col] = df[col] - seasonal_decompose(df[col], model='multiplicative', period=12).seasonal
    a = list(df[col])
    for i in range(1, len(df[col])):
        df[col][i] = a[i]/a[i-1] - 1
        #df[col][i] = np.log(df[col][i])
        
for col in ['Fed_Bonds_10']:
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
    for i in range(1, len(df[col])):
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

for col in ['GDP_(%)_m/m_real_2021', 'Interest_rate_(%)', 'Inflation_m/m_without_seas', 'Fed_Bonds_10', 'real_eff_exchange_rate_index_m/m', 'Brent', 'IMOEX', 'consumption_real_2021', 'M2', 'exp_inf_firms_seas', 'spread_diff']:
    adf_test(df[col], title=col)

df['Date'] = pd.to_datetime(df['Date'])

df.set_index('Date', inplace=True)

df = df.apply(pd.to_numeric, errors='coerce')

with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name = 'Data', index=False)

model_var = VAR(df)
lag_order_result = model_var.select_order(maxlags=8)
#print(lag_order_result.summary())

y = df.values
print(y.shape)
K = y.shape[1]  # количество переменных
p = 8           # лаги

# X: [y_{t-1}, y_{t-2}]
X = []
for t in range(p, len(y)):
    row = np.concatenate([y[t - lag] for lag in range(1, p+1)])
    X.append(row)
X = np.array(X)  # (T-p, p*K)
Y = y[p:]        # (T-p, K)

def estimate_residual_variances(df, variables, ar_lags=1):
    #Оценивает дисперсию остатков AR(1) моделей для каждой переменной
    #Это соответствует "сигма_i^2" в Minnesota prior
    sigma_sq = {}
    for var in variables:
        series = df[var].dropna()
        if len(series) < ar_lags + 10:
            raise ValueError(f"Слишком мало данных для {var}")
        
        model = AutoReg(series, lags=ar_lags, trend='c').fit()
        resid_var = np.var(model.resid)  # дисперсия остатков
        sigma_sq[var] = resid_var
    return sigma_sq

def create_bvar_priors(variables, p, custom_priors_dict, df, 
                                 lambda_0=0.4,     # общая сила априора
                                 lambda_1=1.0,    # убывание по лагам
                                 lambda_self=0.9, 
                                 lambda_cross=0.4 
                                 ):
    """
    Создаёт матрицы mu и sigma для Minnesota prior:
    - sigma_ij = lambda_0 * (sigma_to / sigma_from) / (lag+1)^lambda_1 * (lambda_self или lambda_cross)
    - mu = 0, кроме особых случаев
    """
    K = len(variables)
    total_coeffs = p * K
    mu_matrix = np.zeros((total_coeffs, K))
    sigma_matrix = np.zeros((total_coeffs, K))

    # AR(1) 
    sigma_resid_sq = estimate_residual_variances(df, variables, ar_lags=1)
    sigma_sq = {var: max(val, 1e-6) for var, val in sigma_resid_sq.items()}  # защита от 0

    for lag in range(p):
        lag_factor = 1 / (lag + 1)**lambda_1 

        for j in range(K):  # from_var (лаг переменной j)
            for i in range(K):  # to_var (уравнение переменной i)
                idx = lag * K + j
                from_var = variables[j]
                to_var = variables[i]
                key = (from_var, to_var)

                # Проверяем специфический априор
                if key in custom_priors:
                    mu_base, sigma_base = custom_priors[key]
                    decay = 0.7 ** lag
                    mu_val = mu_base * decay
                    sigma_b = lambda_0 * (sigma_sq[to_var] / sigma_sq[from_var])**0.5
                    sigma_val = sigma_b * sigma_base 
                else:
                    # Общее правило Minnesota prior
                    sigma_base = lambda_0 * (sigma_sq[to_var] / sigma_sq[from_var])**0.5 * lag_factor

                    if i == j:  # собственный лаг
                        mu_val = 0.0  # можно оценить автокорреляцию, но пока 0
                        sigma_val = sigma_base * lambda_self
                    else:  # внешняя переменная
                        mu_val = 0.0
                        sigma_val = sigma_base * lambda_cross

                mu_matrix[idx, i] = mu_val
                sigma_matrix[idx, i] = sigma_val

    return mu_matrix, sigma_matrix

## === Задаём переменные и специфические априоры ===
variables = ['GDP_(%)_m/m_real_2021', 'Interest_rate_(%)', 'Inflation_m/m_without_seas', 'Fed_Bonds_10', 'real_eff_exchange_rate_index_m/m', 'Brent', 'IMOEX', 'consumption_real_2021', 'M2', 'exp_inf_firms_seas', 'spread_diff']

custom_priors = {
    #('Interest_rate_(%)', 'GDP_(%)_m/m_real_2021'): (-0.01, 0.1),
    #('real_eff_exchange_rate_index_m/m', 'GDP_(%)_m/m_real_2021'): (0.1, 0.8),
    ('M2', 'GDP_(%)_m/m_real_2021'): (0.01, 0.1),
    ('GDP_(%)_m/m_real_2021', 'GDP_(%)_m/m_real_2021'): (0.116178, 0.5),
    ('Interest_rate_(%)', 'Interest_rate_(%)'): (0.211324, 0.5),
    ('Inflation_m/m_without_seas', 'Inflation_m/m_without_seas'): (0.307053, 1),
    ('Fed_Bonds_10', 'Fed_Bonds_10'): (0.106080, 0.5),
    ('real_eff_exchange_rate_index_m/m', 'real_eff_exchange_rate_index_m/m'): (0.195712, 1),
    ('Brent', 'Brent'): (0.322764, 1),
    ('IMOEX', 'IMOEX'): (-0.0912668, 0.5),
    ('consumption_real_2021', 'consumption_real_2021'): (0.991977, 2),
    ('M2', 'M2'): (-0.0186251, 0.5),
    ('exp_inf_firms_seas', 'exp_inf_firms_seas'): (0.158409, 0.5),
    ('spread_diff', 'spread_diff'): (0.0274490, 0.5),
    
    #('Interest_rate_(%)', 'GDP_(%)_m/m_real_2021'): (-0.00844966, 0.1),
    #('Inflation_m/m_without_seas', 'GDP_(%)_m/m_real_2021'): (-0.017991123, 0.1),
    #('Fed_Bonds_10', 'GDP_(%)_m/m_real_2021'): (-0.161305291, 1),
    #('real_eff_exchange_rate_index_m/m', 'GDP_(%)_m/m_real_2021'): (0.0140419, 0.1),
    #('Brent', 'GDP_(%)_m/m_real_2021'): (0.028866331, 0.3),
    #('IMOEX', 'GDP_(%)_m/m_real_2021'): (0.011973678, 0.2),
    #('consumption_real_2021', 'GDP_(%)_m/m_real_2021'): (1.178708976, 4),
    #('M2', 'GDP_(%)_m/m_real_2021'): (0.007490262, 0.1),
    #('exp_inf_firms_seas', 'GDP_(%)_m/m_real_2021'): (-0.006737923, 0.1)
    
    #('Interest_rate_(%)', 'GDP_(%)_m/m_real_2021'): (-0.0196110, 0.1),
    #('Inflation_m/m_without_seas', 'GDP_(%)_m/m_real_2021'): (-0.383497, 3),
    #('Fed_Bonds_10', 'GDP_(%)_m/m_real_2021'): (-0.465778, 5),
    #('real_eff_exchange_rate_index_m/m', 'GDP_(%)_m/m_real_2021'): (0.0402469, 0.4),
    #('Brent', 'GDP_(%)_m/m_real_2021'): (0.0429941, 0.4),
    #('IMOEX', 'GDP_(%)_m/m_real_2021'): (0.0231614, 0.2),
    #('consumption_real_2021', 'GDP_(%)_m/m_real_2021'): (0.762241, 4),
    #('M2', 'GDP_(%)_m/m_real_2021'): (0.007490262, 0.1),
    #('exp_inf_firms_seas', 'GDP_(%)_m/m_real_2021'): (0.00923231, 0.1), 
    
    ('Interest_rate_(%)', 'GDP_(%)_m/m_real_2021'): (-0.01, 0.1),
    ('Inflation_m/m_without_seas', 'GDP_(%)_m/m_real_2021'): (-0.01, 3),
    ('Fed_Bonds_10', 'GDP_(%)_m/m_real_2021'): (-0.01, 5),
    ('real_eff_exchange_rate_index_m/m', 'GDP_(%)_m/m_real_2021'): (0.01, 0.4),
    ('Brent', 'GDP_(%)_m/m_real_2021'): (0.01, 0.4),
    ('IMOEX', 'GDP_(%)_m/m_real_2021'): (0.01, 0.2),
    ('consumption_real_2021', 'GDP_(%)_m/m_real_2021'): (0.01, 4),
    ('M2', 'GDP_(%)_m/m_real_2021'): (0.01, 0.1),
    ('exp_inf_firms_seas', 'GDP_(%)_m/m_real_2021'): (0.01, 0.1)   
}
# Задали матрицы характеристик априорного распределения
mu_prior, sigma_prior = create_bvar_priors(
    df=df,
    variables=variables,
    p=8,
    custom_priors_dict=custom_priors,
    lambda_0=0.5,
    lambda_1=1.0,
    lambda_self=1.0,
    lambda_cross=0.2
)


        
eps = 1e-8
total_coeffs = mu_prior.shape[0]  # p*K
Kvars = len(variables)             # = K

# Инициализация (фиктивные наблюдения)
Xprior = np.zeros((total_coeffs * Kvars, total_coeffs))  # каждый ряд — однобитный для конкретного коэф в конкретном уравнении
Yprior = np.zeros((total_coeffs * Kvars, Kvars))
vprior = 0.0 # параметр степеней свободы для обратного вишартова априора

row = 0
for coeff_idx in range(total_coeffs):
    for eq_idx in range(Kvars):
        sd = float(sigma_prior[coeff_idx, eq_idx])
        sd = max(sd, eps) # защита от нулей
        weight = 1.0 / sd

        # ставим вес на ту же позицию в X (коэффициент coeff_idx)
        Xprior[row, coeff_idx] = weight
        # Yprior для уравнения eq_idx: mu / sd
        Yprior[row, eq_idx] = float(mu_prior[coeff_idx, eq_idx]) * weight

        row += 1

# Сконструируем info для var_dummyobsprior
info_vdp = {
    'lags': p,
    'dummyobs': {
        'X': Xprior,
        'Y': Yprior,
        'v': vprior
    }
}

# Основной массив y для var_dummyobsprior — используем те же данные, что у вас уже в Y, но 
# var_dummyobsprior ожидает полную матрицу y (T, N) исходную (до лагирования).
# У нас она есть в переменной y (в начале кода вы делали y = df.values).
y_full = df.values

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
    if "dummyobs" in info: # это есть в информации (построено заранее)
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
            decay = float(mn.get("decay", 0.2))

            # оценка стандартных отклонений остатков AR(1)
            if "sigma" not in mn:
                sigma_data = np.asarray(mn.get("sigma_data", y))
                sigma = np.zeros(N)
                sigma_arlags = int(mn.get("sigma_arlags", max(0, min(P, sigma_data.shape[0] - 3))))
                for n in range(N): #Стандартное отклонение остатков AR(1)
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
                sigma = sigma * float(mn["sigma_factor"])  # type: ignore

            # Prior for coefficients: p(B|Sigma) = N(B0, Sigma ⊗ Q) ? возможно проблема с весами ?
            Winv = (np.arange(1, P + 1, dtype=float) ** decay) #вектор весов для коэффициентов (1/std(B))
            Winv = np.kron(Winv, 1/(sigma / float(mn["tightness"]))) #кронекерово произведение
            Winv = np.concatenate([Winv, (exog_std ** -1) * np.ones(k_exog)])
            Winv_mat = np.diag(Winv)

            B0 = np.zeros((K, N))
            mvector = np.asarray(mn.get("mvector", np.ones(N)))
            mvector = mvector[:N]
            # set means of first lag own coefficients
            for i in range(N):
                # first lag block occupies rows [0:N)
                B0[i, i] = 1

            Yprior = np.vstack([Yprior, Winv_mat @ B0])
            Xprior = np.vstack([Xprior, Winv_mat])

            # Prior for variance: p(Sigma) = IW(Sprior, vprior) априор на ковариационную матрицу ошибок
            sigma_deg = float(mn.get("sigma_deg", N + 2))
            Z = np.diag(sigma * np.sqrt(max(sigma_deg - N - 1, 1e-12)))
            Yprior = np.vstack([Yprior, Z])
            Xprior = np.vstack([Xprior, np.zeros((N, K))])
            vprior += sigma_deg

            if verbosity:
                print("Minnesota prior")
                print(f"Note: for a proper prior need sigma_deg > {N - 1}")
                print(f"Note: for E(Sigma) to exist need sigma_deg > {N + 1}")

        # Sims' dummy observations
        sd = info.get("simsdummy", None)
        if sd is not None:
            ybar = np.mean(y[:P, :], axis=0)
            # oneunitroot
            wP = np.zeros((k_exog,)) if k_exog == 0 else w[P - 1, :]
            oneunitroot = float(sd.get("oneunitroot", 0.0))
            if oneunitroot > 0:
                Xprior = np.vstack([
                    Xprior,
                    np.concatenate([np.tile(ybar, P) * oneunitroot, oneunitroot * wP])
                ])
                Yprior = np.vstack([Yprior, ybar * oneunitroot])
            # oneunitrootc
            oneunitrootc = float(sd.get("oneunitrootc", 0.0))
            if oneunitrootc > 0 and k_exog > 0:
                Xprior = np.vstack([
                    Xprior,
                    np.concatenate([np.zeros(P * N), oneunitrootc * wP])
                ])
                Yprior = np.vstack([Yprior, np.zeros(N)])
            # oneunitrooty
            oneunitrooty = float(sd.get("oneunitrooty", 0.0))
            if oneunitrooty > 0:
                Xprior = np.vstack([
                    Xprior,
                    np.concatenate([np.tile(ybar, P) * oneunitrooty, np.zeros(k_exog)])
                ])
                Yprior = np.vstack([Yprior, ybar * oneunitrooty])
            # nocointegration
            nocointegration = sd.get("nocointegration", 0.0)
            noc_vec = np.full(N, float(nocointegration)) if np.isscalar(nocointegration) else np.asarray(nocointegration).astype(float)
            if noc_vec.size > 0 and np.any(noc_vec > 0):
                temp = np.diag(ybar * noc_vec[:N])
                # If Minnesota mvector is given, select only those with mvector==1
                mn_mvector = np.asarray(info.get("minnesota", {}).get("mvector", np.ones(N)))
                mask = mn_mvector[:N].astype(bool)
                temp = temp[mask, :]
                if temp.size > 0:
                    Xprior = np.vstack([Xprior, np.hstack([np.tile(temp, (1, P)), np.zeros((temp.shape[0], k_exog))])])
                    Yprior = np.vstack([Yprior, temp])
            if verbosity:
                print("Sims dummy prior")

    # Training sample prior добавление обучающей выборки как еще одного априора
    trspl = info.get("trspl", [])
    if trspl:
        for ts in trspl:
            ytr = ts.get("y")
            Tsubj = float(ts.get("Tsubj", 0))
            wtr = ts.get("w")
            if ytr is None or Tsubj <= 0:
                continue
            Ytr, Xtr = _varlags(np.asarray(ytr), P)
            if wtr is not None:
                Xtr = np.hstack([Xtr, np.asarray(wtr)[P:, :]])
            scale = np.sqrt(Tsubj / max(1.0, Ytr.shape[0]))
            Ytr *= scale
            Xtr *= scale
            vprior += Tsubj
            Yprior = np.vstack([Yprior, Ytr])
            Xprior = np.vstack([Xprior, Xtr])
            if verbosity:
                print("Training sample prior")

    # Store prior
    prior = {
        "info": info,
        "Yprior": Yprior,
        "Xprior": Xprior,
        "vprior": vprior,
    }

    # Actual data matrices
    Y, X = _varlags(y, P)
    if k_exog > 0:
        X = np.hstack([X, w[P:, :]])
    Y0 = y[:P, :]

    # Stack with priors объединение реальных и фиктивных данных
    Yst = np.vstack([Yprior, Y]) if Yprior.size else Y
    Xst = np.vstack([Xprior, X]) if Xprior.size else X

    # Marginal likelihood
    # prior маргинальное правдоподобие учитывает неопределенность оценки параметров относительно выбранной модели
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
    # stacked
    logdetXtXst = _logdet(Xst.T @ Xst)
    Bst, *_ = np.linalg.lstsq(Xst, Yst, rcond=None)
    Ust = Yst - Xst @ Bst
    Sst = Ust.T @ Ust
    logdetSst = _logdet(Sst)

    if verbosity:
        print(f"Prior degrees of freedom of Sigma: {vprior}")
        if vprior <= (N - 1):
            print("Prior for Sigma is improper!")
            #логарифмическая маргинальная плотность
    logdensy = (-0.5 * N * T * np.log(np.pi)
                + 0.5 * N * (logdetXtXprior - logdetXtXst)
                + _multgammaln(N, 0.5 * (T + vprior)) - _multgammaln(N, 0.5 * vprior)
                + 0.5 * vprior * logdetSprior - 0.5 * (T + vprior) * logdetSst)

    # Posterior moments
    beta = Bst 
    # posterior mean of Sigma
    sigma = Sst / max(T + vprior - N - 1, 1.0)

    beta_draws = None
    sigma_draws = None
    
    #генерация МСМС-выборок из апостериорного наблюдения
    if n_draws and n_draws > 0:
        df = int(round(T + vprior))
        # Compute chol(inv(Xst'Xst)) using Cholesky of Xst'Xst
        XtX = Xst.T @ Xst
        cF, lower = cho_factor(XtX)
        # chol(inv(A)) = inv(chol(A))^T for MATLAB upper-tri; here we use lower flag
        chol_inv = solve_triangular(cF, np.eye(cF.shape[0]), lower=lower)
        XtXinv_chol = chol_inv.T  # shape (K,K)

        try:
            Sst_chol = np.linalg.cholesky(Sst)
        except LinAlgError:
            # add small ridge if Sst not PD
            Sst_chol = np.linalg.cholesky(Sst + 1e-10 * np.eye(N))

        beta_draws = np.empty((K, N, n_draws))
        sigma_draws = np.empty((N, N, n_draws))
        rng = np.random.default_rng()
        for d in range(n_draws):
            Z = rng.standard_normal(size=(df, N))
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
        prior=prior,
        logdensy=float(logdensy),
        lags=P,
        beta=beta,
        sigma=sigma,
        beta_draws=beta_draws,
        sigma_draws=sigma_draws,
    )

# Запустим var_dummyobsprior (предполагается, что она в namespace)
print("Запуск var_dummyobsprior (conjugate BVAR с фиктивными наблюдениями)...")
var_res = var_dummyobsprior(y=y_full, w=None, n_draws=1000, info=info_vdp, verbosity=1)

# var_res содержит:
#  - var_res.beta: (K*p, K) апостериорное среднее коэффициентов
#  - var_res.sigma: (K, K) апостериорная средняя ковариационная матрица ошибок
#  - var_res.beta_draws: (K*p, K, n_draws) или None - МСМС выборки коэффициентов
#  - var_res.sigma_draws: (K, K, n_draws) или None

beta_post_mean = var_res.beta
sigma_post_mean = var_res.sigma

# Если не сгенерировались выборки (beta_draws), создадим "псевдо-trace" из среднего
if var_res.beta_draws is None or var_res.sigma_draws is None:
    # создадим стабильный набор выборок вокруг среднего (маленькое случайное разбросы)
    n_draws_used = 500
    beta_draws = np.repeat(beta_post_mean[:, :, None], n_draws_used, axis=2) \
                 + 1e-6 * np.random.standard_normal(size=(beta_post_mean.shape[0],
                                                           beta_post_mean.shape[1],
                                                           n_draws_used))
    sigma_draws = np.repeat(sigma_post_mean[:, :, None], n_draws_used, axis=2) \
                  + 1e-8 * np.random.standard_normal(size=(sigma_post_mean.shape[0],
                                                            sigma_post_mean.shape[1],
                                                            n_draws_used))
    print('ne rabotaet')
else:
    beta_draws = var_res.beta_draws
    sigma_draws = var_res.sigma_draws
    print('vse rabotaet')

# Для совместимости: сделаем beta_samples в форме (n_samples, K*p, K)
n_samples = beta_draws.shape[2]
beta_samples = beta_draws.transpose(2, 0, 1).copy()  # (n_samples, K*p, K)
#print(n_samples)
## ------------------ Визуализация априор vs апостериор (замена PyMC plot) --------------
## Возьмём тот же элемент: var_idx = 1, coeff_idx = 0
#var_idx = 0
#coeff_idx = 1

## Постериорная выборка для конкретного коэффициента: берем по всем сэмплам
#posterior_sample = beta_samples[:, coeff_idx, var_idx]

#plt.figure(figsize=(10, 5))
#prior_mean = mu_prior[coeff_idx, var_idx]
#prior_sigma = sigma_prior[coeff_idx, var_idx]
#x = np.linspace(prior_mean - 4*prior_sigma, prior_mean + 4*prior_sigma, 300)
#plt.plot(x, norm.pdf(x, prior_mean, prior_sigma), label='Априор', color='gray', linewidth=2)

## KDE постериора (простая оценка)
#sns.kdeplot(posterior_sample, label='Апостериор', color='blue')
#plt.axvline(np.mean(posterior_sample), color='blue', linestyle='--', alpha=0.7)
#plt.title(f"Априор vs Апостериор: {variables[coeff_idx]}(-1) → {variables[var_idx]}")
#plt.xlabel("Коэффициент")
#plt.ylabel("Плотность")
#plt.legend()
#plt.grid(True, alpha=0.3)
#plt.show()
df_sigma = pd.DataFrame(sigma_post_mean)
with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
    df_sigma.to_excel(writer, sheet_name = 'sigma', index=False)
# ------------------ compute_irf_bvar (обновлённая версия, использует beta_samples) ---------------
def compute_irf_from_beta_samples(beta_samples, variables, p, shock_var, response_var, horizon=8, sigma_draws=None):
    """
    beta_samples: (n_samples, K*p, K)
    sigma_draws: (K, K, n_samples) or None (not used here)
    Возвращает irfs: (n_samples, horizon)
    """
    n_samples = beta_samples.shape[0]
    K = len(variables)
    shock_idx = variables.index(shock_var)
    response_idx = variables.index(response_var)

    irfs = np.zeros((n_samples, horizon))

    for i in range(n_samples):
        beta_i = beta_samples[i]  # (K*p, K)
        # Companion matrix A_companion (K*p, K*p)
        A_comp = np.zeros((K * p, K * p))
        for lag in range(p):
            start = lag * K
            end = (lag + 1) * K
            A_lag = beta_i[start:end, :].T  # (K, K) — коэффициенты в форме (rows=equations, cols = regressors)
            # В совместимой форме нам нужно поставить блок (K x K) в первые K строк
            A_comp[:K, lag*K : (lag+1)*K] = A_lag

        if p > 1:
            A_comp[K:, :K*(p-1)] = np.eye(K*(p-1))

        # Шок: используем 1 std ошибки переменной (приближённо из sigma_post_mean)
        # Если sigma_draws передан, можно брать по i, иначе берем из sigma_post_mean
        if sigma_draws is not None:
            sigma_i = sigma_draws[:, :, i]
        else:
            sigma_i = sigma_post_mean * 5
        stds = np.sqrt(np.diag(sigma_i))
        shock_vector = np.zeros(K * p)
        shock_vector[shock_idx] = stds[shock_idx]  # шок в текущем периоде (помещаем в блок первых K)

        state = shock_vector.copy()
        for h in range(horizon):
            irfs[i, h] = state[response_idx]
            state = A_comp @ state

    return irfs
    

#def compute_irf_bvar(trace, variables, p, shock_var, response_var, horizon=8):
    #"""
    #Вычисляет импульсные отклики (IRF) для BVAR модели.
    #Шок = 1 стандартное отклонение ошибки переменной shock_var.
    #Исправлена ошибка с формой матрицы.
    #"""
    ## Извлекаем коэффициенты: (chains, draws, K*p, K) → (samples, K*p, K)
    #beta_samples = trace.posterior['beta'].values
    #beta_samples = beta_samples.reshape(-1, beta_samples.shape[-2], beta_samples.shape[-1])
    ## Теперь beta[i] имеет форму (K*p, K)

    ## Извлекаем chol_stds: (chains, draws, K) → (samples, K)
    #try:
        #stds = trace.posterior['chol_stds'].values
        #stds = stds.reshape(-1, stds.shape[-1])
    #except:
        #print("Warning: 'chol_stds' not found. Using std = 1.0 for all variables.")
        #stds = np.ones((beta_samples.shape[0], len(variables)))

    #K = len(variables)
    #n_samples = beta_samples.shape[0]
    #shock_idx = variables.index(shock_var)
    #response_idx = variables.index(response_var)

    #irfs = np.zeros((n_samples, horizon))

    #for i in range(n_samples):
        ## --- Шаг 1: Собираем companion matrix ---
        ## Форма: (K*p, K)
        #beta_i = beta_samples[i]  # (K*p, K)

        ## Создаём companion matrix размера (K*p, K*p)
        #A_companion = np.zeros((K * p, K * p))

        ## Заполняем первые K строк: коэффициенты при лагах
        #for lag in range(p):
            #start_idx = lag * K
            #end_idx = (lag + 1) * K
            ## Коэффициенты при y_{t-lag-1} в уравнениях → столбцы [start_idx:end_idx]
            ## Нам нужна матрица (K, K): beta_i[start_idx:end_idx, :] → (K, K)
            #A_lag = beta_i[start_idx:end_idx, :]  # (K, K) — коэффициенты при y_{t-lag-1}
            #A_companion[:K, lag*K : (lag+1)*K] = A_lag  # встаёт как блок

        ## Остальная часть companion matrix — сдвиг
        #if p > 1:
            #A_companion[K:, :K*(p-1)] = np.eye(K*(p-1))  # сдвиг состояния вперёд

        ## --- Шаг 2: Шок в 1 std ---
        #shock_vector = np.zeros(K)
        #shock_vector[shock_idx] = stds[i, shock_idx]  # шок = 1 std

        ## Начальное состояние: [y_t, y_{t-1}, ..., y_{t-p+1}] → вектор (K*p,)
        #state = np.zeros(K * p)
        #state[:K] = shock_vector  # шок в текущем периоде

        ## --- Шаг 3: Импульсный отклик ---
        #for h in range(horizon):
            #irfs[i, h] = state[response_idx]
            #state = A_companion @ state

    #return irfs

# Пример: шок по ВВП → реакция инфляции
#irf_samples = compute_irf_bvar(trace, variables, p, shock_var='Interest_rate_(%)', response_var='GDP_(%)_m/m_real_2021', horizon=8)

#median_irf = np.median(irf_samples, axis=0)
#lower_irf, upper_irf = np.percentile(irf_samples, [5, 95], axis=0)

#plt.figure(figsize=(10, 5))
#plt.plot(median_irf, marker='o', label='Медианный отклик', color='darkblue')
#plt.fill_between(range(8), lower_irf, upper_irf, alpha=0.3, color='blue', label='90% интервал')
#plt.axhline(0, color='black', linewidth=0.8)
#plt.title("Функция импульсного отклика: шок ставки → ввп")
#plt.xlabel("Лаги")
#plt.ylabel("Отклик")
#plt.legend()
#plt.grid(True, alpha=0.3)
#plt.show()


shock_variables = [
    'Interest_rate_(%)',
    'Inflation_m/m_without_seas',
    'Fed_Bonds_10',
    'real_eff_exchange_rate_index_m/m',
    'Brent',
    'IMOEX',
    'consumption_real_2021',
    'M2',
    'exp_inf_firms_seas',
    'spread_diff'
]

# Настройка графика: сетка 3x3 (или автоматически)
n_shocks = len(shock_variables)
cols = 3
rows = (n_shocks + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(14, 10))
axes = axes.flatten()  # Упрощаем доступ к осям 

response_var = 'GDP_(%)_m/m_real_2021'
horizon = 8

# Для каждой переменной строим IRF
for idx, shock_var in enumerate(shock_variables):
    ax = axes[idx]
    
    # Вычисляем IRF 
    irf_samples = compute_irf_from_beta_samples(beta_samples, variables, p, 
                                   shock_var=shock_var, 
                                   response_var=response_var, 
                                   horizon=horizon)
    
    median_irf = np.median(irf_samples, axis=0)
    print(f'{shock_variables[idx]}, {np.sum(median_irf)}')
    lower_irf, upper_irf = np.percentile(irf_samples, [5, 95], axis=0)
    
    # График
    ax.plot(median_irf, marker='o', color='darkblue', linewidth=1.2, label='Медианный отклик')
    ax.fill_between(range(horizon), lower_irf, upper_irf, alpha=0.3, color='blue', label='90% ДИ')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f"{shock_var}", fontsize=10)
    ax.set_xlabel("Лаги")
    ax.set_ylabel("Отклик ВВП")
    ax.grid(True, alpha=0.3)
    ax.legend()

# Убираем пустые подграфики, если есть
for idx in range(n_shocks, len(axes)):
    fig.delaxes(axes[idx])

# Оптимизируем расположение
plt.tight_layout()
plt.suptitle("Функции импульсного отклика: Реакция ВВП на шоки различных переменных",
             fontsize=16, y=1.02)
plt.show()