```@meta
CurrentModule = DynamicPanelModels
```

# DynamicPanelModels.jl

**DynamicPanelModels.jl** is a Julia package for estimating linear dynamic
panel data models — panel regressions with a lagged dependent variable — via
Generalized Method of Moments (GMM). It provides the three standard
estimators from the dynamic panel literature:

- **[`DifferenceGMM`](@ref)** — Arellano & Bond (1991)
- **[`SystemGMM`](@ref)** — Blundell & Bond (1998)
- **[`AndersonHsiao`](@ref)** — Anderson & Hsiao (1981), as a simple IV baseline

## Why this exists

Panel data with a lagged dependent variable
(``y_{it} = \alpha y_{i,t-1} + x_{it}'\beta + \eta_i + \varepsilon_{it}``)
cannot be consistently estimated with OLS or the standard fixed-effects
(within) estimator: differencing or demeaning to remove the individual effect
``\eta_i`` mechanically correlates the transformed lagged-dependent-variable
regressor with the transformed error term (Nickell, 1981). GMM estimators
built around instrumenting that lagged term — Difference
GMM and System GMM — are the standard fix, widely used in applied
macro/micro panel work (e.g. growth regressions, firm investment, labor
dynamics) wherever `T` is short and `N` is large.

DynamicPanelModels.jl implements that estimation pipeline end to end:
correct instrument construction (including the subtleties of unbalanced
panels and forward orthogonal deviations), one-step and two-step GMM with
cluster-robust and Windmeijer (2005) finite-sample-corrected standard errors,
and the diagnostic tests (Sargan/Hansen, Arellano-Bond AR tests) needed to
check whether the identifying assumptions actually hold on your data —
because a dynamic panel GMM estimate without those checks is not
trustworthy on its own.

## Key features

- **Three estimators** in one consistent interface: `fit(DifferenceGMM(), df; ...)`.
- **Correct inference**: one-step/two-step GMM, cluster-robust and
  Windmeijer-corrected standard errors — not just plug-in asymptotic SEs.
- **Diagnostics built in**: Sargan/Hansen J-test, AR(1)/AR(2) serial
  correlation tests, instrument-proliferation check, Wald test, and a
  difference-in-Hansen test for nested instrument sets, all via
  [`diagnose`](@ref).
- **Practical controls**: instrument collapsing and `min_lag`/`max_lag`
  limits (`xtabond2`-style) for large-`T` panels where the instrument count
  can otherwise explode; forward orthogonal deviations for unbalanced panels.
- **Ecosystem-native**: accepts any [Tables.jl](https://github.com/JuliaData/Tables.jl)
  source, supports both string and StatsModels `@formula` syntax, and fitted
  models implement the standard StatsAPI (`coef`, `vcov`, `stderror`,
  `confint`, …) plus RecipesBase diagnostic plots.

## Quick example

```julia
using DynamicPanelModels, DataFrames

# df must be in long format with :id and :time columns
model = fit(DifferenceGMM(robust=true), df;
            formula = "y ~ lag(y) + x1",
            id_col = :id,
            time_col = :time,
            exog = ["x1"])

println(model)      # Stata-like summary table
diagnose(model)      # Sargan, AR(1)/AR(2), normality tests
```

## Where to go next

- **[Getting Started](@ref)** — installation, input data format, a full
  worked example, and the methodology behind each estimator.
- **[API Reference](@ref)** — every exported function, organized by category.

## Scope

This package covers *linear* dynamic panel models (a continuous dependent
variable with a lagged-dependent-variable regressor). Nonlinear dynamic panel
models — binary choice or count data with a lagged dependent variable — are a
distinct econometric problem without a widely-used reference implementation
to validate against, and are out of scope here.
