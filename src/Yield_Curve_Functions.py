import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import jarque_bera
from datetime import datetime as dt
from pandas_datareader import data as web
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D 



FRED_SERIES = {
    "DGS1MO": 1/12,
    "DGS3MO": 3/12,
    "DGS6MO": 6/12,
    "DGS1":   1.0,
    "DGS2":   2.0,
    "DGS3":   3.0,
    "DGS5":   5.0,
    "DGS7":   7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}

START_DATE = "2005-01-01"
END_DATE = dt.today().strftime("%Y-%m-%d")

OWID_IPCC_FILE = "/Users/emanuelemigliaccio/Climate-Risk-Project/data/owid_ipcc_scenarios.csv"


TARGETS = ["beta0", "beta1", "beta2"]

SIM_END_YEAR = 2100

MATURITIES = np.array([1, 2, 3, 5, 7, 10, 20, 30], dtype=float)

np.random.seed(42)

TARGETS = ["beta0", "beta1", "beta2"]

# Macro-economic and climate variable blocks
BLOCKS = {
    "macro": [
        "GDP",
        "GDP per capita",
        "Economic consumption per capita",
        "Population",
    ],
    "climate_physical": [
        "Temperature",
        "CO2 concentration",
        "Methane concentration",
        "Nitrous oxide concentration",
        "Radiative forcing",
    ],
    "transition": [
        "Carbon price",
        "CO2 emissions per capita",
        "Methane emisisons per capita",   
        "Nitrous oxide emissions per capita",
        "Carbon intensity of economy",
        "Carbon intensity of energy",
        "Primary energy (%, fossil)",
        "Primary energy (%, coal)",
        "Primary energy (%, gas)",
    ],
    "energy_use": [
        "Primary energy",
        "Final energy",
        "Electricity per capita",
        "Final energy per capita",
    ],
}


#PCA components per block
N_COMPONENTS = {
    "macro": 1,
    "climate_physical": 2,
    "transition": 2,
    "energy_use": 2,
}





def ns_loadings(tau, lamb):
    tau = np.asarray(tau, dtype=float)
    x = tau / lamb
    x = np.where(x == 0, 1e-8, x)
    L2 = (1 - np.exp(-x)) / x
    L3 = L2 - np.exp(-x)
    return L2, L3

def nelson_siegel(tau, b0, b1, b2, lamb):
    L2, L3 = ns_loadings(tau, lamb)
    return b0 + b1 * L2 + b2 * L3

def ns_residuals(params, x, y):
    b0, b1, b2, lamb = params
    return y - nelson_siegel(x, b0, b1, b2, lamb)

def fit_ns_to_curve(maturities, yields_row, lamb_init=0.5):
    """
    Fit NS a una singola curva.
    Ritorna (b0, b1, b2, lambda) oppure NaN se dati insufficienti.
    """
    y = np.asarray(yields_row, dtype=float)
    m = np.asarray(maturities, dtype=float)

    mask = ~np.isnan(y)
    if mask.sum() < 4:
        return np.array([np.nan, np.nan, np.nan, np.nan])

    x = m[mask]
    y = y[mask]

    b0_init = np.nanmean(y)
    b1_init = y[0] - b0_init
    b2_init = 0.0
    p0 = np.array([b0_init, b1_init, b2_init, lamb_init])

    bounds_lower = [-5.0, -10.0, -10.0, 0.01]
    bounds_upper = [15.0, 10.0, 10.0, 5.0]

    res = least_squares(
        ns_residuals,
        p0,
        bounds=(bounds_lower, bounds_upper),
        args=(x, y),
        max_nfev=500
    )

    if not res.success:
        return np.array([np.nan, np.nan, np.nan, np.nan])

    return res.x

def load_fred_yields(series, start_date, end_date):
    df = pd.DataFrame()
    for code in series:
        s = web.DataReader(code, "fred", start_date, end_date)
        df[code] = s[code]
    return df


def fit_ns_timeseries(yields_m):
    print("Stimo fattori Nelson-Siegel mensili...")
    maturities = [FRED_SERIES[c] for c in yields_m.columns]
    rows = []
    for date, row in yields_m.iterrows():
        b0, b1, b2, lamb = fit_ns_to_curve(maturities, row.values)
        if not np.any(np.isnan([b0, b1, b2, lamb])):
            rows.append({
                "Date": date,
                "beta0": b0,
                "beta1": b1,
                "beta2": b2,
                "lambda": lamb
            })
    ns = pd.DataFrame(rows).set_index("Date").sort_index()
    return ns

def plot_ns_fit_examples(yields_m, ns_hist, maturities, n_samples=6):
    """Confronta curve FRED vs NS fitted per alcune date casuali."""
    import random
    dates = random.sample(list(ns_hist.index), n_samples)
    dates = sorted(dates)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, date in enumerate(dates):
        ax = axes[i]
        y_obs = yields_m.loc[date].values
        mask = ~np.isnan(y_obs)
        mats = np.array(maturities)[mask]
        y_obs = y_obs[mask]

        b = ns_hist.loc[date, ["beta0", "beta1", "beta2", "lambda"]].values
        y_fit = nelson_siegel(mats, *b)

        ax.plot(mats, y_obs, "o-", label="Osservato", color="tab:blue")
        ax.plot(mats, y_fit, "--", label="Nelson–Siegel", color="tab:orange")
        ax.set_title(date.strftime("%Y-%m"))
        ax.set_xlabel("Maturity (anni)")
        ax.set_ylabel("Yield (%)")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend()
    plt.suptitle("Fit Nelson–Siegel vs dati FRED (campione mensile)", fontsize=14)
    plt.tight_layout()
    plt.show()


def build_macro_monthly(df, scenario, macro_vars, start_year, end_year):
    sub = df[df["Scenario"] == scenario].copy()
    if sub.empty:
        raise ValueError(f"Nessun dato per scenario {scenario} nel CSV.")

    # limitiamo solo agli anni necessari
    sub = sub[(sub["Year"] >= start_year) & (sub["Year"] <= end_year)]
    if sub.empty:
        raise ValueError(f"Nessun dato per {scenario} tra {start_year}-{end_year}.")

    idx = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="M")
    out = pd.DataFrame(index=idx)

    years = sub["Year"].values.astype(float)
    for col in macro_vars:
        if col not in sub.columns:
            raise ValueError(f"Colonna {col} mancante nel CSV.")
        vals = sub[col].values.astype(float)
        out[col] = np.interp(
            np.linspace(start_year, end_year, len(idx)),
            years,
            vals
        )
    return out

def build_block_pca(df_hist, blocks, n_components_dict):
    """
    Costruisce PCA per ciascun blocco di variabili.
    Output:
    - pca_features_hist: DataFrame con le componenti PCA storiche
    - block_scalers: dict {block_name: StandardScaler}
    - block_pcas: dict {block_name: PCA}
    """
    block_scalers = {}
    block_pcas = {}
    block_components_hist = []

    for block_name, cols in blocks.items():
        cols_existing = [c for c in cols if c in df_hist.columns]
        if not cols_existing:
            print(f"[WARN] Blocco '{block_name}' senza colonne presenti, saltato.")
            continue

        X_block = df_hist[cols_existing].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_block)

        n_comp = n_components_dict[block_name]
        pca = PCA(n_components=n_comp)
        Z_block = pca.fit_transform(X_scaled)

        block_scalers[block_name] = scaler
        block_pcas[block_name] = pca

        col_names = [f"{block_name}_PC{i+1}" for i in range(n_comp)]
        block_components_hist.append(
            pd.DataFrame(Z_block, index=df_hist.index, columns=col_names)
        )

    if not block_components_hist:
        raise ValueError("Nessun blocco PCA costruito: controllare BLOCKS/df_hist.")

    pca_features_hist = pd.concat(block_components_hist, axis=1)
    return pca_features_hist, block_scalers, block_pcas


def plot_scree_advanced(block_pcas, BLOCKS, title_suffix="Historical"):
    for block_name in BLOCKS.keys():
        if block_name not in block_pcas:
            continue
        pca = block_pcas[block_name]
        ev = pca.explained_variance_ratio_
        cum_ev = ev.cumsum()
        xs = np.arange(1, len(ev)+1)

        plt.figure(figsize=(8,4))
        plt.bar(xs, ev*100, alpha=0.3, color="skyblue", label="Explained variance")
        plt.plot(xs, cum_ev*100, marker="o", color="steelblue",
                 linewidth=2, label="Cumulative")
        for i, v in enumerate(cum_ev):
            plt.text(xs[i], v*100+1, f"{v*100:.1f}%", ha="center", fontsize=8)

        for thr, style in [(90, "--"), (95, "--"), (99, ":")]:
            plt.axhline(thr, color="#888", linestyle=style, linewidth=1)
            plt.text(1, thr+0.5, f"{thr}%", color="#555", fontsize=8)

        plt.title(f"PCA Explained Variance — {block_name} ({title_suffix})")
        plt.xlabel("Component")
        plt.ylabel("Explained variance (%)")
        plt.ylim(0, 105)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_pca_loadings(block_pcas, BLOCKS):
    for block_name, cols in BLOCKS.items():
        if block_name not in block_pcas:
            continue
        pca = block_pcas[block_name]
        loadings = pd.DataFrame(
            pca.components_,
            columns=cols,
            index=[f"PC{i+1}" for i in range(pca.n_components_)]
        )
        for i in range(pca.n_components_):
            plt.figure(figsize=(10,3))
            plt.bar(loadings.columns, loadings.iloc[i].values)
            plt.xticks(rotation=45, ha="right")
            plt.title(f"Loadings {block_name} – PC{i+1}")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()



def fit_dns_pca_deltas_AR2_stable(ns_hist_sub, pca_features_hist, targets,
                                  rho_max=0.90, alphas=None):
    """
    Model:
        beta_t = a + phi1 * beta_{t-1} + phi2 * beta_{t-2} + delta' * dZ_t + eps_t

    """

  
    data = ns_hist_sub.join(pca_features_hist, how="inner")
    pca_vars = list(pca_features_hist.columns)


    for col in pca_vars:
        data[f"d_{col}"] = data[col].diff()


    for t in targets:
        data[f"{t}_lag1"] = data[t].shift(1)
        data[f"{t}_lag2"] = data[t].shift(2)

    data = data.dropna()

    pca_delta_vars = [f"d_{col}" for col in pca_vars]

    if alphas is None:
        alphas = np.logspace(-4, 4, 21)

    params = {
        "a": {},
        "phi1": {},
        "phi2": {},
        "delta": {},
        "sigma_resid": {},
        "pca_vars": pca_delta_vars,
    }
    models = {}

    Z = data[pca_delta_vars].values
    z_mean = Z.mean(axis=0)

    for t in targets:
        y = data[t].values
        mu_y = y.mean()

        X_lag1 = data[f"{t}_lag1"].values.reshape(-1, 1)
        X_lag2 = data[f"{t}_lag2"].values.reshape(-1, 1)
        X = np.hstack([X_lag1, X_lag2, Z])

        model = RidgeCV(alphas=alphas, cv=5)
        model.fit(X, y)

  
        a_raw = float(model.intercept_)
        phi1_raw = float(model.coef_[0])
        phi2_raw = float(model.coef_[1])
        delta_raw = model.coef_[2:].copy()


        sum_phi = phi1_raw + phi2_raw
        if sum_phi > rho_max:
            scale = rho_max / sum_phi
            phi1 = phi1_raw * scale
            phi2 = phi2_raw * scale
        else:
            phi1 = phi1_raw
            phi2 = phi2_raw


        a_new = mu_y * (1.0 - phi1 - phi2) - float(np.dot(delta_raw, z_mean))

        y_hat = (
            a_new
            + phi1 * X_lag1.ravel()
            + phi2 * X_lag2.ravel()
            + Z @ delta_raw
        )
        resid = y - y_hat
        sigma_resid = resid.std()

        # salvataggio
        params["a"][t] = a_new
        params["phi1"][t] = phi1
        params["phi2"][t] = phi2
        params["delta"][t] = delta_raw
        params["sigma_resid"][t] = sigma_resid

        models[t] = model

        print(
            f"{t}: phi1_raw={phi1_raw:.3f}, phi2_raw={phi2_raw:.3f}, "
            f"phi1={phi1:.3f}, phi2={phi2:.3f}, "
            f"sigma={sigma_resid:.4f}, alpha_ridge={model.alpha_:.4f}"
        )

    return params, models

def diagnose_ar2(ns_hist_sub, pca_features_hist, params, target):
    """
    Residuals models:
        beta_t = a + phi1 * beta_{t-1} + phi2 * beta_{t-2} + ...
    """
    data = ns_hist_sub.join(pca_features_hist, how="inner")
    pca_vars = list(pca_features_hist.columns)


    for col in pca_vars:
        data[f"d_{col}"] = data[col].diff()
    data[f"{target}_lag1"] = data[target].shift(1)
    data[f"{target}_lag2"] = data[target].shift(2)

    data = data.dropna()
    pca_delta_vars = [f"d_{col}" for col in pca_vars]

    y = data[target].values
    X_lag1 = data[f"{target}_lag1"].values.reshape(-1, 1)
    X_lag2 = data[f"{target}_lag2"].values.reshape(-1, 1)
    X_pca = data[pca_delta_vars].values
    X = np.hstack([X_lag1, X_lag2, X_pca])

    a = params["a"][target]
    phi1 = params["phi1"][target]
    phi2 = params["phi2"][target]
    delta = params["delta"][target]

    y_hat = a + phi1 * X_lag1.ravel() + phi2 * X_lag2.ravel() + np.dot(X_pca, delta)
    resid = y - y_hat

    print("="*70)
    print(f"Diagnostica AR(2) – {target}")
    print("="*70)
    print(f"phi1 = {phi1:.4f}, phi2 = {phi2:.4f}")
    print(f"sigma(resid) = {resid.std():.4f}")

    jb = jarque_bera(resid)
    print(f"Jarque-Bera: stat={jb[0]:.3f}, p-value={jb[1]:.3f}")

    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    ax[0].plot(resid)
    ax[0].axhline(0, color="black", lw=1)
    ax[0].set_title(f"{target} – residui AR(2)")
    ax[0].grid(alpha=0.3)

    plot_acf(resid, ax=ax[1])
    ax[1].set_title("ACF residui")

    plot_pacf(resid, ax=ax[2])
    ax[2].set_title("PACF residui")

    plt.tight_layout()
    plt.show()

def transform_scenarios_with_pca(df_scn, blocks, n_components_dict, block_scalers, block_pcas):
    block_components_scn = []

    for block_name, cols in blocks.items():
        if block_name not in block_scalers:
            continue
        cols_existing = [c for c in cols if c in df_scn.columns]
        if not cols_existing:
            continue

        X_block = df_scn[cols_existing].values
        scaler = block_scalers[block_name]
        pca = block_pcas[block_name]

        X_scaled = scaler.transform(X_block)
        Z_block = pca.transform(X_scaled)

        n_comp = n_components_dict[block_name]
        col_names = [f"{block_name}_PC{i+1}" for i in range(n_comp)]
        block_components_scn.append(
            pd.DataFrame(Z_block, index=df_scn.index, columns=col_names)
        )

    if not block_components_scn:
        raise ValueError("Nessun blocco PCA applicabile allo scenario.")

    pca_features_scn = pd.concat(block_components_scn, axis=1)
    return pca_features_scn




def simulate_betas_AR2(ns_hist_sub, pca_features_scn, targets, params,
                       stochastic=False, random_state=None,
                       z_clip=3.0, beta_clip=(-5, 10)):
    """
    Simulate:
        β_t = a + φ1 β_{t-1} + φ2 β_{t-2} + δ' z_t  + ε_t

   
    """

    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    pca_vars = params["pca_vars"]
    Z = pca_features_scn[pca_vars].copy()

    

    last = ns_hist_sub.iloc[-1]
    prev = ns_hist_sub.iloc[-2]

    betas_t1 = {t: float(last[t]) for t in targets}  
    betas_t2 = {t: float(prev[t]) for t in targets}  

    out = {t: [] for t in targets}

    for idx, row in Z.iterrows():
        z_t = row.values

        new_betas = {}

        for t in targets:
            a = params["a"][t]
            phi1 = params["phi1"][t]
            phi2 = params["phi2"][t]
            delta = params["delta"][t]
            sigma = params["sigma_resid"][t]

            mean_t = a + phi1 * betas_t1[t] + phi2 * betas_t2[t] + np.dot(delta, z_t)

            eps = rng.normal(scale=sigma) if stochastic else 0.0

            b_next = mean_t + eps


            new_betas[t] = b_next
            out[t].append(b_next)


        betas_t2 = betas_t1
        betas_t1 = new_betas

    return pd.DataFrame(out, index=Z.index)


def nelson_siegel_yield(maturities, beta0, beta1, beta2, tau=1.0):
    maturities = np.asarray(maturities)
    lam = 1.0 / tau
    x = maturities * lam

    with np.errstate(divide='ignore', invalid='ignore'):
        factor1 = np.where(x == 0, 1.0, (1 - np.exp(-x)) / x)
        factor2 = factor1 - np.exp(-x)

    y = beta0 + beta1 * factor1 + beta2 * factor2
    return y

def make_yield_surface_from_betas(betas_df, maturities, tau=1.0):
    curves = []
    for date, row in betas_df.iterrows():
        b0, b1, b2 = row["beta0"], row["beta1"], row["beta2"]
        y = nelson_siegel_yield(maturities, b0, b1, b2, tau=tau)
        curves.append(pd.Series(y, index=maturities, name=date))
    surface = pd.DataFrame(curves)
    return surface

def plot_yield_surface(surface, title="Yield surface"):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    times = np.arange(len(surface.index))
    maturities = surface.columns.values.astype(float)
    T, M = np.meshgrid(times, maturities, indexing="ij")
    Z = surface.values

    surf = ax.plot_surface(T, M, Z,
                           cmap=cm.viridis,
                           linewidth=0,
                           antialiased=True,
                           alpha=0.9)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Maturity")
    ax.set_zlabel("Yield")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_yield_lines_by_maturity(surface, title="Yield by maturity"):
    plt.figure(figsize=(10,6))
    for m in surface.columns:
        plt.plot(surface.index, surface[m], label=f"{m}y", linewidth=2)
    plt.xlabel("Time index")
    plt.ylabel("Yield")
    plt.title(title)
    plt.legend(ncol=3)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_yield_heatmap(surface, title="Yield heatmap"):
    plt.figure(figsize=(10,6))
    plt.imshow(surface.T, aspect="auto", origin="lower",
               cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Yield")
    plt.xlabel("Time index")
    plt.ylabel("Maturity")
    plt.title(title)
    plt.tight_layout()
    plt.show()