# Methodology

This document states every modelling choice, the equation that implements it and the reason
for it. Numbers quoted here come from the auto-generated [results summary](../results/README.md).
The code follows the same structure:

- `climate_risk.term_structure` covers Section 3.
- `climate_risk.credit` covers Section 4.
- `climate_risk.portfolio` covers Section 5.

## 1. Design principles

The project asks a single question. How does a climate-transition pathway change the
long-horizon value of corporate debt, through the risk-free curve and through credit spreads?
Four principles drive the design.

1. **Estimate what history identifies, calibrate what it does not.** US data contain rich
   information on yield-curve dynamics, but almost none on the effect of a global
   decarbonisation pathway. Climate variables have moved smoothly and monotonically over the
   sample, so a regression of yields on them is spurious. Therefore:
   - the dynamics (factor persistence, shock covariance) are **estimated**;
   - the mapping from scenario macro variables to the *long-run anchor* of the curve is
     **calibrated** from theory and published estimates, and is reported with sensitivities.
2. **Use the IAM scenarios for what they are designed for: relative comparisons.**
   - Rates: every scenario is measured against a reference pathway.
   - Credit: every policy pathway is measured against the baseline of the *same* SSP, and
     therefore of the same integrated assessment model.
   - Both are harmonised to the observed state at the jump-off date, with the same
     convergence rule.
3. **One consistent pricing logic.** The scenario component of yields, both in the
   projections and in the repricing of bonds, follows the expectations hypothesis. Credit
   uses one structural model (Merton with fixed face value) for the calibration and for the
   shocks.
4. **Keep pricing and risk measures in their own probability measure.**
   - Spreads and bond values are risk-neutral (Q).
   - Portfolio loss distributions are physical (P).

## 2. Data

| Input | Source | Frequency, sample | Use |
|---|---|---|---|
| Zero-coupon Treasury yields, 1–30y | Gürkaynak, Sack & Wright (2007), Federal Reserve | month-end, 1986-01 → 2026-09 | Nelson–Siegel factors |
| Natural rate r\*, trend growth | Holston, Laubach & Williams (2023), NY Fed, one-sided | quarterly, lagged one quarter | endpoint |
| 10-year expected inflation π\* | Federal Reserve Bank of Cleveland (FRED `EXPINF10YR`) | monthly | endpoint |
| SSP scenarios (26 pathways, world) | IIASA SSP Database (Riahi et al., 2017) via Our World in Data | 2005, 2010, 2020 … 2100 | scenario drivers, sector output |
| CDS par spreads, 1–10y | licensed panel, 661 issuers (not redistributed) | daily, 2015-01 → 2021-09 | credit calibration, dependence |
| Moody's Baa – 10y Treasury | FRED `BAA10Y` | daily | public fallback ("demo mode") |

The processed extracts the models read are versioned in `data/processed/`, which pins the data
vintage behind the published results. `climate-risk download` refreshes them. See
[`data/README.md`](../data/README.md) for licences.

## 3. The term structure under SSP scenarios

### 3.1 Cross-section: Nelson–Siegel with one decay parameter

Following Diebold & Li (2006), the zero-coupon yield of maturity $\tau$ (years) in month $t$ is

$$
y_t(\tau) = \beta_{1t} + \beta_{2t}\,\frac{1-e^{-\lambda\tau}}{\lambda\tau}
+ \beta_{3t}\left[\frac{1-e^{-\lambda\tau}}{\lambda\tau}-e^{-\lambda\tau}\right] + \varepsilon_t(\tau).
$$

The decay $\lambda$ is held **constant over time**. The loadings are then fixed, the betas are
obtained month by month by OLS, and they form a coherent time series of level, slope and
curvature. Version v0.1 let $\lambda$ vary every day during estimation, fixed it to a different
value in the projection, and so mixed incompatible betas.

$\lambda$ is estimated by profiling the pooled sum of squared errors over all months and the
15 maturities. The profile is **flat and bimodal**:

- **Global minimum, $\lambda \approx 0.12$ (8.4 bp).** The curvature hump sits at 15 years and
  fits the long-end hump of the GSW curve. The level and slope then lose their interpretation:
  the correlation of the level with the 30y yield falls to 0.88.
- **Local minimum, $\hat\lambda = 0.473$ (8.7 bp).** The curvature peaks at 3.8 years, the
  correlation of the level with the 30y yield is 0.99, and the correlation of the slope with
  the 1y–30y spread is 0.97.

Because the scenario channel acts on the *level* (Section 3.3), the search is restricted to
decays whose curvature peak lies between 1.2 and 7.2 years, which keeps a medium-term third
factor as in Diebold & Li. The figure `01_lambda_profile` shows the whole profile.

### 3.2 Dynamics: a shifting endpoint

Over 1986–2026 the level of yields trends down with the natural rate. A constant-mean VAR
reads this trend as near-unit-root persistence: the level eigenvalue is 0.992, a half-life of
about 7 years. It also pulls long-horizon forecasts towards the 1980s sample mean. Following
Bauer & Rudebusch (2020), the level is split into a slow-moving endpoint and a stationary cycle:

$$
\boldsymbol\beta_t = \mathbf x_t + \mathbf d_t,\qquad
\mathbf d_t = (i^{\ast}_t, 0, 0)',\qquad
i^{\ast}_t = r^{\ast}_{t} + \pi^{\ast}_t,
$$

$$
\mathbf x_t = \mathbf c + \Phi\,\mathbf x_{t-1} + \boldsymbol\eta_t,\qquad
\boldsymbol\eta_t \sim N(\mathbf 0, \Sigma).
$$

- **Endpoint.** $r^{\ast}$ is the one-sided (filtered) HLW estimate of the *previous* quarter,
  so the state carries no look-ahead. The HLW parameters, however, are estimated on the
  current-vintage sample: the series is not real-time. $\pi^{\ast}$ is the Cleveland Fed
  10-year expected inflation.
- **Cycle.** The VAR(1) on the cycle is estimated by OLS on 489 months.
- **Persistence.** All eigenvalues are inside the unit circle. The half-lives are 62, 12 and
  8 months, against 86, 33 and 6 months for the constant-mean VAR.
- **Mean.** The cycle mean of the level, $(I-\Phi)^{-1}\mathbf c$, is a level premium of
  0.9 pp over $i^{\ast}$.

### 3.3 Scenario-conditional endpoint

In projection, the deterministic component becomes scenario-specific, $\mathbf d(s,t)$. Two
channels connect a pathway $s$ to the curve.

**Growth (Ramsey/HLW) channel, on the level.** HLW model $r^{\ast} = c\,g + z$, where $g$ is
trend growth, and estimate $c = 1.113$ for the United States. We set $\theta = c$ and write

$$
r^{\ast}(s,t) = r^{\ast}_{T_0} + \theta\,\big[\tilde g(s,t) - g_{\text{ref}}(t)\big],
\qquad i^{\ast}(s,t) = r^{\ast}(s,t) + \pi^{\ast}_{T_0}.
$$

- $\tilde g(s,t)$ is the scenario's world GDP-per-capita growth: the derivative of a natural
  cubic spline through $\ln$ GDP per capita.
- It is harmonised to the reference growth at the jump-off $T_0$ (Section 3.4).
- $g_{\text{ref}}$ is the growth of the reference pathway, SSP2-Baseline.
- **The reference pathway keeps today's $r^{\ast}$.** In HLW, $r^{\ast}$ is a random walk, so
  its best forecast is its current value. Using absolute world growth instead would import
  the end of emerging-market convergence into US $r^{\ast}$.
- Only *differences* between pathways move the anchor. The driver is global growth because
  the decline of natural rates is common to advanced economies (Holston, Laubach & Williams,
  2017).
- **Two caveats.**
  - HLW's $g$ is US trend growth of total output, while the scenarios provide world GDP per
    capita. $\theta$ is therefore applied to an analogous, not identical, object.
  - IMAGE, the SSP1 marker model, treats GDP as exogenous: GDP is identical across the SSP1
    policy variants. The growth channel of SSP1 policies is zero by construction.

**Carbon-price ("greenflation") channel, on the slope.** The carbon bill as a share of GDP is

$$
c(s,t) = \frac{p^{\text{CO}_2}(s,t)\; E^{\text{gross}}(s,t)}{\text{GDP}(s,t)}.
$$

- **Tax base.** $E^{\text{gross}}$ is gross energy-and-industry CO₂: net emissions plus BECCS
  removals, floored at zero. Payments for negative emissions are fiscal transfers, not a cost
  passed on to the consumer basket.
- **Transmission.** With pass-through $\psi$, the price level moves with $c$, so inflation is
  higher by $\pi_c = \psi\,\dot c$ while the bill rises and lower while it falls.
  Long-run expectations stay anchored at $\pi^{\ast}_{T_0}$, while the short rate responds by
  $\kappa\,\pi_c$ (Taylor principle, $\kappa = 1.5$). The rule is therefore
  $\mathbf d(s,t) = (i^{\ast}(s,t),\ \kappa\pi_c(s,t),\ 0)'$, and the curvature factor has no
  scenario channel.
- **Where the impulse comes from.** In the 2017 vintage, the stringent pathways already price
  carbon at the jump-off: in SSP2-1.9, $c$ is 1.4% of GDP in 2026 and still 1.4% in 2050.
  Observed carbon-pricing revenue is about 0.1% of world GDP (World Bank, 2024). Harmonising
  $c$ to the observed level therefore implies a *catch-up*, from 0.1% to the pathway's level
  by the convergence year $T_c$. The harmonisation is multiplicative (Section 3.4): the
  catch-up starts smoothly from today's level, and the IAM's own ramp in the 2020s is scaled
  down instead of being carried over. The inflation impulse is mostly that catch-up, so its
  timing is an assumption and is reported with sensitivities:
  - the 10-year repricing of SSP2-1.9 is +16, +6, +2 and +1 bp for $T_c$ = 2040, 2050, 2060
    and 2070;
  - it is −4.5 bp if the central bank looks through carbon-price inflation ($\kappa=0$).
- **Two objects, one anchor.** The observed anchor is revenue, covering about a quarter of
  emissions at heterogeneous prices and measured at market exchange rates. $c$ is a uniform
  shadow price on all emissions over PPP GDP. Given the gap between 0.1% and 1.4%, the
  approximation does not change the order of magnitude.

**Why not regress yields on climate variables?** Version v0.1 did this on 2005–2024 values
*of the SSP2-4.5 scenario*, interpolated linearly between 2005, 2010 and 2020. The regressors
were piecewise constant, so seven of them carried two or three pieces of information, and the
coefficients were then applied at a different frequency and scale. Published cross-country
estimates do not fill the gap either. In the panel of Collender et al. (2022), +50 pp of
renewable share maps to −5 pp on yields: an extrapolation far outside the data. The structural
mapping above is transparent, and every assumption is varied in
`results/tables/rates_sensitivity.csv`.

### 3.4 Harmonisation

The SSP vintage (2017) lets pathways diverge from 2020, which has not happened. All scenario
variables are harmonised to the observed state at $T_0$ and rejoin the native IAM path by the
convergence year $T_c = 2050$, with the same weight

$$
w(t) = \tfrac12\left[1 + \cos\!\left(\pi\,\tfrac{t-T_0}{T_c-T_0}\right)\right]
\quad\text{on }[T_0, T_c],\qquad w = 0 \text{ afterwards}.
$$

The cosine taper is a C¹ version of the linear convergence of Gidden et al. (2018), so
derived rates (growth, inflation) have no jump at $T_c$. Two variants of the rule are used,
following Gidden et al.:

- **Additive offset, for growth rates:**
  $\tilde v(t) = v(t) + [v^{\text{obs}}_{T_0} - v(T_0)]\,w(t)$.
- **Ratio, for positive quantities:**
  $\tilde v(t) = v(t)\,\big[1 + \big(v^{\text{obs}}_{T_0}/v(T_0) - 1\big)\,w(t)\big]$.
  It applies to the carbon-cost share $c$ (observed value 0.1%) and to the policy-to-baseline
  output ratio $1+u$ of Section 4.3 (observed value 1: no divergence yet). It keeps the
  quantity positive, and it scales the IAM's rate of change at $T_0$ by
  $v^{\text{obs}}_{T_0}/v(T_0)$, whereas the offset would carry it over unchanged. In the
  1.5 °C pathways (SSP1/2/5-1.9) the carbon bill at $T_0$ is 14 to 27 times today's, and 6 to
  10 times in the 2.6 pathways. The offset would then start the projection with the IAM's
  2020s ramp: in SSP1-1.9, carbon-price inflation of about 0.5 pp a year from the first month.
  Pathways without a carbon price keep today's 0.1% throughout; the offset is used only for a
  pathway whose carbon price starts after $T_0$.

In both cases the pathway starts from today's state and catches up with the IAM by $T_c$.
"No catch-up" variants, in which the 2020–2026 divergence is never recovered, are reported as
sensitivities.

### 3.5 Monte Carlo and the scenario overlay

The cycle is simulated monthly to December 2100: 2,000 Gaussian paths from the observed state
of September 2026. The same draws are shared by every scenario (common random numbers). The
scenario enters yields through an **expectations-hypothesis overlay** with unchanged term
premia:

$$
y_s(t,\tau) = \Lambda(\tau)'\mathbf x_t + o_s(t,\tau),\qquad
o_s(t,\tau) = \frac1\tau\int_t^{t+\tau}\big[d_1(s,u) + d_2(s,u)\big]\,du ,
$$

where $d_1 + d_2$ is the scenario's deterministic shift of the instantaneous short rate
$\beta_1+\beta_2$.

- **Reference pathway.** Its endpoint is constant, so $o_s = i^{\ast}_{T_0}$ at every
  maturity. This is exactly the estimation set-up, where the endpoint is a martingale.
- **Other pathways.** A $\tau$-year yield anticipates the whole scenario path over the life of
  the bond. The same overlay prices the repricing of bonds when a pathway is revealed
  (Section 4.5), so projections and repricing are mutually consistent.
- **Quantiles.** Every quantile of $y_s$ is the quantile of the cycle yield shifted by
  $o_s(t,\tau)$. Scenario differences are exact and carry no simulation noise. Reported: the
  mean, median and 25/75 and 5/95% bands per year, scenario and maturity.
- **Bands are conditional on the endpoint path.** In HLW, the quarterly innovation of $r^{\ast}$
  has a standard deviation of 0.19 pp. Taken literally as a random walk, this would add a
  standard deviation of about 3.3 pp by 2100. Uncertainty about the anchor is instead
  represented by the scenario ensemble: by 2050 the SSP narratives span 2.1–4.8% for the
  10-year yield.

### 3.6 Validation

Pseudo-out-of-sample forecasts are made from every month since January 2000 on expanding
windows, at horizons of 1, 6, 12, 24 and 60 months. Four models are compared:

- random walk;
- DNS-AR(1) (Diebold–Li);
- DNS-VAR(1);
- the shifting-endpoint VAR, whose endpoint is held at its current value.

**Test.** Accuracy is measured by RMSE and the Diebold–Mariano test.
- The long-run variance uses the first $h-1$ autocovariances with equal weights, the original
  DM estimator, with the Harvey–Leybourne–Newbold correction.
- If that estimate is not positive, Bartlett (Newey–West) weights are used instead.

**Results.**
- **Random walk.** It is hard to beat, a well-documented result after 2000. In 27 of 75
  model/horizon/maturity cells it is *significantly* more accurate at 5%: 21 against the
  constant-mean DNS models (horizons up to 24 months) and 6 against the shifting-endpoint VAR
  (1 and 6 months only).
- **Shifting endpoint.** It is the best DNS variant at long horizons. At 60 months it lowers
  the 10y relative RMSE from 1.38 (AR(1)) and 1.12 (VAR) to 1.05. On the 2-year yield it
  beats the random walk at 24 and 60 months (0.98 and 0.97).
- **Significance.** Only one improvement is significant: the 1-year yield at 60 months
  ($p = 0.03$). At that horizon, the 22 years of usable origins contain about four
  non-overlapping windows, so the test has little power.
- **Scope.** The model is used for conditional scenario analysis, not for market timing.

**Non-real-time inputs.** The backtest uses the full-sample $\hat\lambda$, a cross-sectional
shape parameter that Diebold & Li fix ex ante, and the current-vintage $r^{\ast}$. Estimated
on 1986–1999, $\lambda$ reaches the lower bound of 0.25. The relative RMSEs of the
shifting-endpoint model then move by at most 0.02 at 24–60 months and by up to 0.1 at one
month, through the cross-sectional fit.

## 4. Climate-adjusted credit spreads

### 4.1 Risk-neutral default probabilities from CDS

For each issuer, the monthly quotes of the pre-COVID window 2015–2019 are summarised by their
median. A piecewise-constant hazard $\lambda_k$ on $(T_{k-1},T_k]$ is bootstrapped tenor by
tenor (O'Kane & Turnbull, 2003) so that

$$
s_k \sum_{j}\Delta_j\Big[P(t_j)Q(t_j) + \tfrac12 P(\bar t_j)\big(Q(t_{j-1})-Q(t_j)\big)\Big]
= (1-R)\sum_j P(\bar t_j)\big(Q(t_{j-1})-Q(t_j)\big).
$$

- Premiums are quarterly, with accrual on default.
- $R = 40\%$.
- $P$ is the window-average Nelson–Siegel Treasury curve and $Q$ the survival function.
- A quote that would require a negative forward hazard is rejected as a static arbitrage.

**Data quality.** It is enforced explicitly:
- issuers quoted in less than 90% of months are dropped;
- so are issuers whose 5y quote is unchanged in more than 25% of weeks, which are stale
  composite marks (Santos, Chugoku Electric);
- issuers in distress are not in the sector map. PG&E filed for bankruptcy in January 2019.

**Sector reference curves.** They are the cross-issuer medians of three groups:
- oil & gas producers, 19 issuers;
- generation-owning utilities, 25 issuers;
- a low-carbon control group of technology, pharmaceutical and telecom firms, 21 issuers.

### 4.2 Merton asset volatility

Under Q with a deterministic short rate and fixed face value $D$, Merton (1974) gives

$$
q(T) = \Phi\!\left(-\,\frac{a + T\,y(T) - \tfrac12\sigma^2 T}{\sigma\sqrt T}\right),\qquad a=\ln(V_0/D).
$$

$(a,\sigma)$ are fitted by least squares in probit space to the bootstrapped term structure.
The fit gives $\sigma_A$ = 0.18 (oil & gas), 0.17 (utilities) and 0.13 (control). Three
remarks:

- **Identification.** $\sigma_A$ is identified through the risk-neutral drift term
  $T\,y(T)-\tfrac12\sigma^2T$, that is, under fixed face value. The same assumption is
  therefore kept in the shocks (Section 4.4). Under a stationary-leverage drift the fit has no
  interior optimum.
- **Fit quality.** The fit is approximate. The probit RMSE is about 0.2: fitted PDs are a third
  of market PDs at one year and twice them at three years, the classic difficulty of Merton at
  short maturities. The implied leverage is not plausible either: 0.57–0.66, highest for the
  control group.
- **Use.** Only $\sigma_A$ is used afterwards; the baseline default probabilities are the
  market ones. Since the shock scales with $1/\sigma_A$, $\sigma_A$ ×0.75 (×1.25) moves the oil &
  gas 10-year climate spread from 59 bp to 83 bp (46 bp).

### 4.3 The Battiston–Monasterolo climate-policy shock

For a transition from the baseline B to the policy P of the same SSP (same IAM), the output
of sector $S$ changes by $u_S(t) = Y_S(P,t)/Y_S(B,t) - 1$ (BM 2020, eq. 4). Three
climate-policy-relevant sectors are taken from the IAM output:

- primary oil and gas supply;
- fossil electricity: coal, gas and oil, CCS included;
- low-carbon electricity: the remainder.

An issuer with business-line weights $w_S$, measured in B, receives the asset shock
$\xi = \chi\sum_S w_S \tilde u_S$ (BM, Section 3.3), with $\tilde u$ harmonised as in
Section 3.4:

- oil & gas: $w_{\text{PrOilGas}} = 1$;
- utilities: the baseline power mix at $T_0$, which is 64% fossil in SSP2, 66% in SSP1 and
  77% in SSP5;
- control: no exposure.

The elasticity $\chi$ is 0.5. BM use 1 and report empirical values of 0.2–0.6, so results are
shown for χ = 0.2, 0.5 and 1.

### 4.4 Climate-adjusted PDs and the climate spread

"Priced today" means priced in the first projected month, $v_0$ = October 2026. Let
$\xi(T)=\chi\sum_S w_S\,\tilde u_S(v_0+T)$ be the shock realised by the maturity of a
$T$-year bond, and $\Delta y(T)$ the repricing of the $T$-year Treasury yield of Section 4.5.
In the lognormal Merton model with fixed face value, the two move the default threshold
additively, which is the lognormal version of BM eqs. 11–14 plus the drift effect:

$$
q_P(T) = \Phi\!\left(\Phi^{-1}\big(q_B(T)\big) - \frac{\ln(1+\xi(T)) + T\,\Delta y(T)}{\sigma_A\sqrt T}\right),
\qquad
s(q,T) = -\frac{\ln(1-q\,\mathrm{LGD})}{T},
\qquad
\Delta s(T) = s(q_P,T) - s(q_B,T).
$$

The spread $s$ is the BM zero-coupon convention (eqs. 7–8). Beyond 10 years the baseline
hazard is extrapolated flat.

- **Drift effect.** It is small. Higher risk-free yields lower risk-neutral PDs: about −2.5 bp
  on a 10-year oil & gas spread and −2 bp for the control group in SSP2-1.9. It is reported
  separately (`rate_effect_bp`) and can be switched off.
- **Delayed revelation.** The transition is under way from $T_0$ but priced only at a later
  date $v$, each January to 2050. The shock becomes $\chi\sum_S w_S\,\tilde u_S(v+T)$, with the
  risk-free repricing at $v$: the "cost of waiting" (`10_delayed_revelation`).

### 4.5 Joint repricing: risk-free channel vs credit channel

At revelation, the Treasury curve reprices by the gap between the scenario overlays of
Section 3.5:

$$
\Delta y(T) = o_P(v,T) - o_B(v,T) = \frac1T\int_v^{v+T} \big(\mathbb E_P[r_u]-\mathbb E_B[r_u]\big)\,du,
$$

where $v$ is the revelation date ($v_0$ when the transition is priced today). The value of a
zero-coupon corporate bond, $Z=e^{-(y_f+s)T}$, then changes by

$$
\frac{Z_P}{Z_B}-1 = \underbrace{\left(e^{-\Delta y\, T}-1\right)}_{\text{risk-free}}
+ \underbrace{\left(e^{-\Delta s\,T}-1\right)}_{\text{credit}} + \text{interaction},
$$

an exact decomposition.

- **SSP2-1.9.** The 10-year yield rises by 6.4 bp, because carbon-price inflation outweighs
  the slightly lower $r^{\ast}$. The risk-free channel costs a 10-year bond 0.6%:
  - about one tenth of the loss of an oil & gas bond (−6.4%);
  - for utilities it offsets the small credit gain, so the total is about zero (+0.1%);
  - for low-carbon issuers it is the whole loss: −0.64%, of which the lower risk-neutral
    default probabilities give back 0.19 pp (−0.45% in total).
- **Sign.** The sign of the channel depends on monetary policy. If the central bank looks
  through carbon-price inflation, yields fall by 4.5 bp instead.
- **Long bonds.** At 30 years the channel nets out to almost zero.

## 5. Portfolio losses

### 5.1 Measures

Horizon PDs bootstrapped from CDS are risk-neutral. For the loss distribution they are
converted to physical PDs by scaling the cumulative hazard:

$$
q^{\mathbb P} = 1 - \left(1 - q^{\mathbb Q}\right)^{1/k},\qquad k = 2.4,
$$

- **Ratio.** $k$ is the risk-neutral/actual intensity ratio estimated by Driessen (2005) for
  BBB issuers; his estimates are 1.8 for AA and 2.6 for A.
- **Reduced form.** A constant ratio is a simplification. In the Merton model the P/Q relation
  would depend on the asset Sharpe ratio and on the horizon.
- **Shocks.** The climate shock and the risk-free repricing move the Merton threshold, and are
  applied identically under both measures (constant asset risk premium).
- **Reporting.** Results are reported under both measures.

### 5.2 Dependence

- **Kendall's τ.** It is computed for every pair of issuers on weekly log-changes of 5y CDS
  spreads, 260 weeks. In a structural model the spread is a decreasing function of asset
  value, and τ is invariant to monotone transformations, so it measures asset dependence.
- **Correlation.** $\rho=\sin(\pi\tau/2)$ (Lindskog, McNeil & Schmock, 2003), averaged within
  and between sectors into a positive-definite block matrix:

  | | Low-carbon | Utilities | Oil & gas |
  |---|---|---|---|
  | **Low-carbon** | 0.31 | | |
  | **Utilities** | 0.35 | 0.46 | |
  | **Oil & gas** | 0.31 | 0.41 | 0.50 |

- **Degrees of freedom.** The t-copula's are estimated by maximum pseudo-likelihood on rank
  uniforms (Demarta & McNeil, 2005): $\hat\nu = 14.4$.
- **Caveats and benchmark.**
  - The weekly changes are not GARCH-filtered.
  - Weekly dependence is applied to a five-year default horizon.
  - CDS co-movements also contain common risk-premium shocks and tend to overstate default
    correlation.
  - A Gaussian copula with the Basel IRB asset correlation (BCBS 2006, §272) is therefore also
    reported, as a benchmark.

### 5.3 Loss distribution

- **Portfolio.** It is equally weighted. Issuer $i$ defaults before the horizon when
  $U_i \le q_i$, with $U$ the copula draws.
- **Simulation.** 200,000 draws are made, shared across scenarios.
- **Risk measures.** VaR and Expected Shortfall at 99% follow Acerbi & Tasche (2002), which
  is coherent for discrete losses.
- **Scenario uncertainty.** It is a mixture of the baseline and the adverse pathway with
  probability $p$ (BM 2020, Section 3.9).
- **Check.** A unit test validates the simulation against Vasicek's large-portfolio quantile.
- **Loss definition.** Losses are default losses only (hold to maturity); mark-to-market
  effects are covered by Section 4.5.

## 6. Parameters

| Symbol | Value | Source | Sensitivity reported |
|---|---|---|---|
| $\lambda$ | 0.473 /yr | estimated (profile, restricted) | full profile; pre-2000 estimate |
| $\theta$ | 1.113 | HLW (2023) estimate of $c$ | 0.5, 1.5 |
| $\kappa$ | 1.5 | Taylor (1993) | 0 (central bank looks through) |
| $\psi$ | 1 | full pass-through (upper bound) | via $\kappa$ |
| $T_c$ | 2050 | harmonisation convergence | 2040, 2060, 2070; no catch-up (credit) |
| $R$ | 40% | ISDA convention | — |
| $\chi$ | 0.5 | BM (2020): 1; empirical 0.2–0.6 | 0.2, 1.0 |
| $\sigma_A$ | 0.13–0.18 | Merton fit to CDS curves | ×0.75, ×1.25 |
| rate drift in PDs | on | fixed-face-value Merton | off |
| $k$ | 2.4 | Driessen (2005), BBB | risk-neutral results ($k=1$) |
| VaR/ES level | 99% | — | — |

## 7. Limitations and extensions

- **Scenario vintage.** The SSP marker quantifications date from 2017 and are world-level, and
  their stringent pathways start pricing carbon in 2020. That is why the harmonisation, and
  the catch-up it implies, matter (Section 3.3). NGFS Phase V scenarios provide
  country-level, more recent pathways, including NiGEM-modelled interest rates, and are the
  natural extension. Their download requires a free IIASA registration.
- **Arbitrage-freeness.** The DNS model is not arbitrage-free. The AFNS correction of
  Christensen, Diebold & Rudebusch (2011) is a convexity term that is small below 30 years.
  Term premia are held constant, both in the overlay and in the repricing, and
  climate-uncertainty premia are not modelled.
- **Zero lower bound.** Gaussian dynamics allow negative short rates in the lower tail of the
  long-run distribution.
- **Growth channel.** HLW's elasticity of US $r^{\ast}$ to US trend growth is applied to world
  per-capita growth differentials. For SSP1 the IAM keeps GDP exogenous, so the growth channel
  is zero there.
- **Shock structure.** The BM shock uses static business-line weights and the output gap at
  maturity. It does not use the present value of the gaps after maturity, and it has no capex
  reallocation and no adaptation. The representative utility owns the market's generation
  mix. Energy-intensive industry and transport would need sector output or firm emissions
  that the OWID extract does not provide.
- **Merton.** The fit is used only for $\sigma_A$. Implied leverage is not plausible, and the
  climate spreads scale with $1/\sigma_A$ (Section 4.2).
- **Baseline credit.** The 2015–2019 baseline credit quality is assumed stationary and is
  reused for revelations up to 2050.
- **Physical risk.** It is out of scope. The bankruptcy of PG&E in January 2019, driven by
  wildfire liabilities, is the reminder that physical risk can dominate for some issuers.
- **Discounting currency.** Discounting uses the US Treasury curve for all issuers. The
  currency of the CDS contracts is not in the data. This is a second-order effect on hazards.

## References

See [`references/references.bib`](../references/references.bib).
