# DynamicPanelModels.jl

[![CI](https://github.com/MichalS16/DynamicPanelModels.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/MichalS16/DynamicPanelModels.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/MichalS16/DynamicPanelModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MichalS16/DynamicPanelModels.jl)
[![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://MichalS16.github.io/DynamicPanelModels.jl/dev/)
[![Julia](https://img.shields.io/badge/Julia-1.12+-9558B2?logo=julia)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Panel data with a lagged dependent variable
($y_{it} = \alpha y_{i,t-1} + x_{it}'\beta + \eta_i + \varepsilon_{it}$)
cannot be consistently estimated with OLS or the standard fixed-effects
(within) estimator — removing the individual effect $\eta_i$ by differencing
or demeaning mechanically correlates the transformed lagged-dependent-variable
regressor with the transformed error (Nickell, 1981). DynamicPanelModels.jl
provides the standard GMM fix for this: the
Arellano-Bond (Difference GMM), Blundell-Bond (System GMM), and
Anderson-Hsiao estimators, with correct instrument construction, robust and
Windmeijer-corrected inference, and the diagnostic tests needed to check
whether the identifying assumptions actually hold before trusting the
results.

> This package covers *linear* dynamic panel models (a continuous dependent
> variable with a lagged-dependent-variable regressor). Nonlinear dynamic
> panel models — binary choice (logit/probit) or count data (Poisson/negative
> binomial) with a lagged dependent variable — are a distinct econometric
> problem (the fixed effect cannot be differenced away, and consistent
> estimators such as Honoré-Kyriazidou (2000) have no widely-used reference
> implementation to validate against) and are out of scope here.

The package targets performance on large-$N$ datasets through sparse matrix
operations, correct inference via robust and Windmeijer-corrected standard
errors, and a suite of diagnostic tools and visualizations for checking model
validity.

## Features

Difference GMM (Arellano-Bond, 1991) removes fixed effects by
first-differencing and supports one-step and two-step estimation, robust
standard errors, and instrument collapsing for large instrument sets. System
GMM (Blundell-Bond, 1998) jointly estimates the differenced and level
equations, offering efficiency gains for persistent series. Anderson-Hsiao
(1981) is provided as a simple IV baseline for comparison.

Standard errors are one-step or two-step, cluster-robust, with the Windmeijer
(2005) finite-sample correction available for the two-step case. Regressors
declared via `exog` are used as their own ("IV-style") GMM instruments,
following standard dynamic panel practice (Roodman, 2009), which improves
identification and efficiency for strictly exogenous covariates.

Diagnostics include the Sargan/Hansen J-test for overidentifying restrictions,
Arellano-Bond AR(1)/AR(2) tests for serial correlation, an instrument
proliferation check, a Wald test for linear restrictions, a Jarque-Bera
normality test, a pseudo-$R^2$, and a difference-in-Hansen (C) test for
comparing nested instrument sets. Diagnostic plots (residuals, fitted vs.
actual, Q-Q, and a combined dashboard) are available via RecipesBase.

The package integrates with the wider ecosystem: any
[Tables.jl](https://github.com/JuliaData/Tables.jl)-compatible input is
accepted (not only `DataFrame`s), formulas can be given as strings or as
StatsModels `@formula`s, and fitted models implement the standard StatsAPI
accessors (`coef`, `vcov`, `stderror`, `residuals`, `fitted`, `predict`,
`confint`, `nobs`, `dof`, `formula`).

The formula syntax supports whitelisted transforms (`log`, `log10`, `log2`,
`exp`, `sqrt`, `abs`) on either side of the formula and nested in lags
(`log(y) ~ lag(log(y)) + log(x)`), explicit and ranged lags (`lag(x, 2)`,
`lag(x, 0:1)`), and automatic period dummies via `time_effects=true`.
Instrument sets can be controlled with forward orthogonal deviations
(`transform=:fod`, better for unbalanced panels), `xtabond2`-style lag limits
(`min_lag`/`max_lag`), instrument collapsing (`collapse`), and automatic
dropping of collinear instruments.

## Installation

This package is currently unregistered and requires Julia 1.12 or higher.
Install it directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/MichalS16/DynamicPanelModels.jl")
```

## Quick start

```julia
using DynamicPanelModels
using DataFrames
using Plots

# df must be in long format with :id and :time columns, e.g.
# | id: 1 | time: 2020 | y: 5.2 | x1: 0.5 |

# x1 is declared exogenous via `exog`, so it is additionally used as its own
# ("IV-style") instrument, following standard GMM dynamic panel practice.
model = fit(DifferenceGMM(robust=true), df;
            formula = "y ~ lag(y) + x1",
            id_col = :id,
            time_col = :time,
            exog = ["x1"])

println(model)
diagnose(model)
plot(model)
```

See the [Getting Started](https://MichalS16.github.io/DynamicPanelModels.jl/dev/guide/)
guide for a full walkthrough.

## Key functions

| Component | Description |
| :--- | :--- |
| `fit(Estimator, df; ...)` | Fits the specified GMM estimator to the provided panel data. Key arguments: `formula`, `id_col`, `time_col`, `exog`. |
| `exog = ["x1", ...]` | Names of RHS regressors (as they appear in `formula`) that are strictly exogenous; each is additionally used as its own GMM instrument. Defaults to none — non-lagged regressors are otherwise only weakly identified through incidental correlation with the `y`-lag instruments. |
| `time_effects = true` | Adds automatically-generated period dummies (T−2, exogenous) to absorb common time shocks. |
| `DifferenceGMM(robust=true)` | Arellano-Bond (1991). Settings: `robust`, `steps`, `windmeijer`. |
| `SystemGMM(robust=true)` | Blundell-Bond (1998), for persistent series ($\rho \approx 1$). Settings: `robust`, `steps`, `windmeijer`. |
| `AndersonHsiao()` | Anderson-Hsiao (1981) IV baseline (no settings). |
| `diagnose(model)` | Runs Sargan, AR, and normality tests together. |
| `sargan_test(model)` | Tests instrument validity (overidentifying restrictions). |
| `ar_test(model, order)` | Arellano-Bond test for serial correlation of order 1, 2, …; cached after `diagnose`. |
| `check_proliferation(model)` | Checks whether instrument count exceeds group count. |
| `wald_test(model, R, r)` | Wald test for linear restrictions $R\beta = r$. |
| `diff_hansen_test(restricted, unrestricted)` | Difference-in-Hansen test for a nested instrument subset (e.g. the same model with and without `exog`). |
| `plot(model)` / `plot(model, :residuals)` / `:fitted` / `:qq` / `:histogram` | Diagnostic dashboard or individual plots. |
| `coef`, `vcov`, `stderror`, `confint`, `residuals`, `fitted`, `nobs`, `ngroups`, `ninstruments` | Standard result accessors. |

## Current limitations

First differences (`:fd`) and forward orthogonal deviations (`:fod`,
Arellano-Bover) are supported for Difference GMM; System GMM currently
supports first differences only. Unbalanced panels are generally supported,
but highly sparse datasets may need manual preprocessing to avoid estimation
errors or excessive data loss. Estimation is restricted to linear dynamic
panel models with continuous dependent variables (see the scope note above).
The Blundell-Bond level-equation instruments are valid only under the
mean-stationarity assumption (the deviation of the initial condition from its
long-run mean is uncorrelated with that long-run mean); this is not
automatically tested, so inspect the Sargan/Hansen diagnostics before trusting
System GMM results on persistent series.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
development workflow and the [Code of Conduct](CODE_OF_CONDUCT.md).

## References

Nickell, S. (1981). Biases in dynamic models with fixed effects.
*Econometrica*, 49(6), 1417-1426.

Arellano, M., & Bond, S. (1991). Some tests of specification for panel data:
Monte Carlo evidence and an application to employment equations. *The Review
of Economic Studies*, 58(2), 277-297.

Blundell, R., & Bond, S. (1998). Initial conditions and moment restrictions in
dynamic panel data models. *Journal of Econometrics*, 87(1), 115-143.

Windmeijer, F. (2005). A finite sample correction for the variance of linear
efficient two-step GMM estimators. *Journal of Econometrics*, 126(1), 25-51.

Roodman, D. (2009). How to do xtabond2: An introduction to difference and
system GMM in Stata. *The Stata Journal*, 9(1), 86-136.

Anderson, T. W., & Hsiao, C. (1981). Estimation of dynamic models with error
components. *Journal of the American Statistical Association*, 76(375),
598-606.

Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel
Data* (2nd ed.). MIT Press.
