# DynamicPanelModels package

[![CI](https://github.com/MichalS16/DynamicPanelModels/actions/workflows/CI.yml/badge.svg)](https://github.com/MichalS16/DynamicPanelModels/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/MichalS16/DynamicPanelModels/branch/main/graph/badge.svg)](https://codecov.io/gh/MichalS16/DynamicPanelModels)
[![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://MichalS16.github.io/DynamicPanelModels/dev/)
[![Julia](https://img.shields.io/badge/Julia-1.12+-9558B2?logo=julia)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## DynamicPanelModels.jl

**DynamicPanelModels.jl** is a Julia package for **linear dynamic panel data models** estimated via Generalized Method of Moments (GMM). It provides implementations of the Arellano-Bond (Difference GMM) and Blundell-Bond (System GMM) estimators.

> **Scope**: this package covers *linear* dynamic panel models (a continuous dependent variable with a lagged-dependent-variable regressor). Nonlinear dynamic panel models — binary choice (logit/probit) or count data (Poisson/negative binomial) with a lagged dependent variable — are a distinct econometric problem (the fixed effect cannot be differenced away, and consistent estimators such as Honoré-Kyriazidou (2000) have no widely-used reference implementation to validate against) and are out of scope for this package.

The package focuses on performance with large-$N$ datasets through sparse matrix operations, ensures correct inference using robust standard errors (including Windmeijer finite-sample corrections), and offers a rich suite of diagnostic tools and visualization capabilities to verify model validity.

## ✨ Features

- **Dynamic Panel Estimators**:
  - **Difference GMM (Arellano-Bond)**: Implements the Arellano-Bond (1991) estimator by first-differencing to remove fixed effects; supports one-step and two-step estimation, robust standard errors, and instrument collapsing for large instrument sets.
  - **System GMM (Blundell-Bond)**: Implements the Blundell-Bond (1998) estimator by jointly estimating differenced and level equations; offers efficiency gains for persistent time series and includes options for Windmeijer finite-sample correction.
  - **Anderson-Hsiao IV**: Provides the Anderson-Hsiao (1981) instrumental variable estimator as a baseline for comparison, using lagged levels as instruments for differenced equations.
- **Robust Inference**: Implements one-step and two-step estimators, featuring cluster-robust standard errors and the Windmeijer finite-sample correction to ensure valid inference.
- **Exogenous Regressor Instrumenting**: Regressors declared via `exog` are used as their own ("IV-style") GMM instruments, following standard dynamic panel practice (Roodman, 2009), improving identification and efficiency for strictly exogenous covariates.
- **Comprehensive Diagnostics**:
  - **Sargan/Hansen J-test**: Validates the instrument set by testing overidentifying restrictions ($H_0$: instruments are exogenous/valid).
  - **Arellano-Bond AR Tests**: Detects serial correlation in the differenced errors; reports both $AR(1)$ (expected due to first-differencing) and $AR(2)$ (indicates model misspecification) statistics.
  - **Instrument Proliferation Checks**: Automatically warns when the instrument count exceeds the number of groups/individuals to prevent overfitting and bias.
  - **Wald Test**: Enables hypothesis testing for linear restrictions on model parameters (e.g., testing if specific coefficients equal zero).
  - **Jarque-Bera Test**: Assesses the normality of model residuals based on skewness and kurtosis.
  - **Pseudo-R²**: Provides a measure of goodness-of-fit based on the correlation between observed and fitted values.
  - **Difference-in-Hansen (C) Test**: A Durbin-Wu-Hausman-style test for the validity of a nested instrument subset, comparing two models fit with the same estimator on the same sample.
- **Visualization**: Built-in plots for residuals, fitted vs. actual values, $Q-Q$ plots, and diagnostic dashboards.
- **StatsAPI Integration**: Fully compatible with standard Julia functions including `coef`, `vcov`, `stderror`, `residuals`, `fitted`, `predict`, `confint`, `nobs`, `dof`, and `formula`.
- **Ecosystem Interoperability**: Accepts any [Tables.jl](https://github.com/JuliaData/Tables.jl)-compatible input (not only `DataFrame`s) and both string formulas and StatsModels `@formula` (e.g. `@formula(y ~ lag(y) + x)`).
- **Flexible Formula Syntax**: Whitelisted transforms (`log`, `log10`, `log2`, `exp`, `sqrt`, `abs`) on either side and nested in lags (`log(y) ~ lag(log(y)) + log(x)`); explicit and ranged lags (`lag(x, 2)`, `lag(x, 0:1)`); and automatic period dummies via `time_effects=true`.
- **Instrument Control**: Forward orthogonal deviations (`transform=:fod`) for unbalanced panels, `xtabond2`-style lag limits (`min_lag`/`max_lag`), instrument collapsing (`collapse`), and automatic dropping of collinear instruments.

## 📦 Installation

This package is currently unregistered and requires **Julia 1.12** or higher.

You can install the latest version directly from GitHub using the `Pkg`:

```julia
using Pkg
Pkg.add(url="https://github.com/MichalS16/DynamicPanelModels")
```

## 🚀 Quick Start

Here is a complete workflow example demonstrating estimation, diagnostics, and visualization.

```julia
# Necessary packages
using DynamicPanelModels
using DataFrames
using Plots

# 1. Load Data - Input DataFrame 'df' must be in Long Format with :id and :time columns.
# (Load your dataset into 'df' here), example: | id: 1 | time: 2020 | y: 5.2 | x1: 0.5 |

# 2. Estimate Model (Difference GMM) - Model: y_it = ρ * y_{i,t-1} + β * x1_it + η_i + ε_it
# 'x1' is declared exogenous via `exog`, so it is additionally used as its own
# ("IV-style") instrument, in line with standard GMM dynamic panel practice.
model = fit(DifferenceGMM(robust=true), df;
            formula = "y ~ lag(y) + x1",
            id_col = :id,
            time_col = :time,
            exog = ["x1"])

# 3. Inspect Results - Summary Output
println(model)

# 4. Run Diagnostics - Automatically runs Sargan, AR(1), and AR(2) tests
diagnose(model)

# 5. Visualize - Generates a diagnostic dashboard (Residuals, Q-Q, Fitted vs Actual)
plot(model)
```

## 🛠️ Key Functions & Data API

| Component | Description |
| :--- | :--- |
| **Input Data** | |
| `df` | Input `DataFrame` in **Long Format** (must have ID and Time columns). |
| **Model Estimation** | |
| `model = fit(Estimator, df; ...)` | **Model Estimation**. Fits the specified GMM estimator to the provided panel data. Key Args: `formula`, `id_col`, `time_col`, `exog`. |
| `exog = ["x1", ...]` | Names of RHS regressors (as they appear in `formula`) that are strictly exogenous; each is additionally used as its own GMM instrument. Defaults to none (all non-lagged regressors are otherwise only weakly identified through incidental correlation with the `y`-lag instruments). |
| `time_effects = true` | Include automatically-generated period dummies (T−2, exogenous) to absorb common time shocks. |
| Formula terms | `lag(v)`, `lag(v, k)`, `lag(v, a:b)` for lags; `log/log10/log2/exp/sqrt/abs` transforms, e.g. `log(y) ~ lag(log(y)) + log(x)`. |
| Estimators | |
| `DifferenceGMM(robust=true)` | **Arellano-Bond (1991)**. Best for standard dynamic panels. Settings: `robust`, `steps`, `windmeijer`. |
| `SystemGMM(robust=true)` | **Blundell-Bond (1998)**. Best for persistent time series ($\rho \approx 1$). Settings: `robust`, `steps`, `windmeijer`. |
| `AndersonHsiao()` | **Anderson-Hsiao (1981)**. Simple baseline IV estimator (no settings). |
| **Diagnostics** | |
| `diagnose(model)` | Runs a full suite of diagnostic tests (Sargan, AR, Normality). |
| `sargan_test(model)` | Tests validity of instruments (Overidentification). |
| `ar_test(model, order, id, time)` | Arellano-Bond test for serial correlation of residuals (order 1, 2, …). After it is computed (or via `diagnose`), `ar_test(model, order)` returns the cached result. |
| `check_proliferation(model)` | Checks if instrument count > group count ($L > N$). |
| `wald_test(model, R, r)` | Performs Wald test for linear restrictions ($R\beta = r$). Key Args: `R`, `r`. |
| `jarque_bera_test(model)` | Tests normality of residuals. |
| `goodness_of_fit(model)` | Calculates Pseudo-R². |
| `diff_hansen_test(restricted, unrestricted)` | Difference-in-Hansen (C statistic) test for the validity of a nested instrument subset (e.g. same model with/without `exog`); a Durbin-Wu-Hausman-style check for GMM instruments. |
| **Visualization** | |
| `plot(model)` | Generates a diagnostic dashboard (4 subplots). |
| `plot(model, :residuals)` | Plots standardized residuals over time. |
| `plot(model, :fitted)` | Plots fitted values vs. actual observed values. |
| `plot(model, :qq)` | Plots $Q-Q$ plot for normality check. |
| `plot(model, :histogram)` | Plots histogram of standardized residuals. |
| **Stats Accessors** | |
| `coef(model)` | Returns estimated coefficients. |
| `vcov(model)` | Returns variance-covariance matrix. |
| `stderror(model)` | Returns robust (Windmeijer) standard errors. |
| `confint(model)` | Returns confidence intervals (default 95%). |
| `residuals(model)` | Returns model residuals vector. |
| `fitted(model)` / `predict(model)` | Returns fitted values vector. |
| `nobs(model)` | Returns total number of observations used. |
| `ngroups(model)` | Returns number of unique individuals/groups. |
| `ninstruments(model)` | Returns number of instruments used. |

## ❗ Current Limitations

- **Transformation Methods**: First differences (`:fd`) and forward orthogonal deviations (`:fod`, Arellano-Bover) are supported for Difference GMM; System GMM currently supports first differences only.
- **Panel Structure**: While unbalanced panels are generally supported, highly sparse datasets may require manual preprocessing to avoid estimation errors or excessive data loss.
- **Model Scope**: Estimation is restricted to linear dynamic panel models with continuous dependent variables (see the Scope note above). Non-linear specifications (e.g., binary choice or count data) are not supported.
- **System GMM Assumptions**: The Blundell-Bond level-equation instruments are valid only under the mean-stationarity assumption (the deviation of the initial condition from its long-run mean is uncorrelated with that long-run mean). This is not automatically tested; users should inspect the Sargan/Hansen diagnostics before trusting System GMM results on persistent series.

## 🤝 Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development
workflow (running tests, formatting, building docs) and the
[Code of Conduct](CODE_OF_CONDUCT.md).

## 📚 References

- **Arellano, M., & Bond, S. (1991).** Some tests of specification for panel data: Monte Carlo evidence and an application to employment equations. *The Review of Economic Studies*, 58(2), 277-297.
- **Blundell, R., & Bond, S. (1998).** Initial conditions and moment restrictions in dynamic panel data models. *Journal of Econometrics*, 87(1), 115-143.
- **Windmeijer, F. (2005).** A finite sample correction for the variance of linear efficient two-step GMM estimators. *Journal of Econometrics*, 126(1), 25-51.
- **Roodman, D. (2009).** How to do xtabond2: An introduction to difference and system GMM in Stata. *The Stata Journal*, 9(1), 86-136.
- **Anderson, T. W., & Hsiao, C. (1981).** Estimation of dynamic models with error components. *Journal of the American Statistical Association*, 76(375), 598-606.
- **Wooldridge, J. M. (2010).** *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.
