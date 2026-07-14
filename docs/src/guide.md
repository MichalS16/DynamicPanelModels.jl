```@meta
CurrentModule = DynamicPanelModels
```

# Getting Started

## Installation

The package is currently unregistered and can be installed directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/MichalS16/DynamicPanelModels.jl")
```

## Input data

Panel data must be in long format, with one row per individual-period
observation and separate ID and time columns:

| id | time | y   | x1  |
|:--:|:----:|:---:|:---:|
| 1  | 2020 | 5.2 | 0.5 |
| 1  | 2021 | 5.6 | 0.6 |
| 2  | 2020 | 3.1 | 0.2 |

Any [Tables.jl](https://github.com/JuliaData/Tables.jl)-compatible source is
accepted, so a `DataFrame`, a `CSV.File`, or any other Tables.jl source will work.

## Fitting a model

```julia
using DynamicPanelModels
using DataFrames

model = fit(DifferenceGMM(robust=true), df;
            formula = "y ~ lag(y) + x1",
            id_col = :id,
            time_col = :time,
            exog = ["x1"])
```

Three estimators are available: [`DifferenceGMM`](@ref) (Arellano-Bond, 1991),
[`SystemGMM`](@ref) (Blundell-Bond, 1998), and [`AndersonHsiao`](@ref) (1981).
Regressors passed via `exog` are additionally used as their own GMM
instruments; without it, non-lagged regressors are only weakly identified
through their correlation with the lagged-dependent-variable instruments.
Setting `time_effects=true` adds period dummies to absorb common time shocks,
and `transform=:fod` switches `DifferenceGMM` from first differences to
forward orthogonal deviations, which handles unbalanced panels better.

## Methodology

All three estimators start from the same fixed-effects AR(1) model,
``y_{it} = \alpha y_{i,t-1} + x_{it}'\beta + \eta_i + \varepsilon_{it}``, with
``\eta_i`` an unobserved individual effect correlated with ``y_{i,t-1}`` (so
OLS/within estimation is biased — Nickell, 1981). Differencing removes
``\eta_i``:
``\Delta y_{it} = \alpha \Delta y_{i,t-1} + \Delta x_{it}'\beta + \Delta \varepsilon_{it}``.

- **Difference GMM** ([`DifferenceGMM`](@ref), Arellano & Bond 1991) instruments
  ``\Delta y_{i,t-1}`` with all valid lagged levels
  ``y_{i,t-2}, y_{i,t-3}, \dots``, relying only on no serial correlation in
  ``\varepsilon_{it}``. It performs poorly when ``\alpha`` is close to 1 or
  the ratio ``\mathrm{var}(\eta_i)/\mathrm{var}(\varepsilon_{it})`` is large,
  because lagged levels become weak predictors of ``\Delta y_{i,t-1}`` in
  that regime.
- **System GMM** ([`SystemGMM`](@ref), Blundell & Bond 1998) adds the levels
  equation back in, instrumented with *lagged differences*
  ``\Delta y_{i,t-1}, \Delta x_{i,t-1}`` (not lagged levels — a common point
  of confusion, since simplified treatments sometimes state the
  level-equation moment condition symbolically as
  ``E[(y_{it} - y_{i,t-1}) y_{i,t-1}] = 0``, which is actually the standard
  first-differenced moment condition rewritten in levels, not the level
  equation's own instrument set). This requires the extra mean-stationarity
  assumption ``E[\eta_i \Delta y_{it}] = 0`` and improves efficiency exactly
  where Difference GMM struggles (``\alpha`` near 1).
- **Anderson-Hsiao** ([`AndersonHsiao`](@ref), 1981) instruments
  ``\Delta y_{i,t-1}`` with a single lagged level ``y_{i,t-2}``, giving a
  consistent but less-efficient IV baseline (it is the ``T=2``-instrument
  special case that Arellano-Bond generalizes).

Standard errors follow Windmeijer (2005): the naive two-step GMM covariance
understates variability because it ignores the estimation error in the
first-step weighting matrix; the correction restores accurate finite-sample
inference. See [`diagnose`](@ref) for the accompanying Sargan/Hansen and AR
tests that check the instrument and serial-correlation assumptions each
estimator relies on.

## Inspecting results

The fitted model supports the usual StatsAPI accessors:

```julia
println(model)      # Stata-like summary table
coef(model)         # coefficient vector
stderror(model)     # standard errors (Windmeijer-corrected, if applicable)
confint(model)      # confidence intervals
```

## Diagnostics

```julia
diagnose(model)     # Sargan, AR(1), AR(2), and normality tests in one call
sargan_test(model)  # instrument validity (overidentifying restrictions)
ar_test(model, 2)   # serial correlation in the differenced errors
```

A significant AR(1) statistic is expected after first-differencing; a
significant AR(2) statistic instead points to model misspecification.

## Visualization

```julia
using Plots
plot(model)              # four-panel diagnostic dashboard
plot(model, :residuals)  # standardized residuals over time
```

The [API Reference](@ref) lists every exported function.
