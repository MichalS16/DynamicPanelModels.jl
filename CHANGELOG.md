# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] — 2026-07-08

### Added

- Difference GMM (Arellano-Bond, 1991) and System GMM (Blundell-Bond, 1998)
  estimators, plus Anderson-Hsiao (1981) as a baseline IV estimator; one-step and
  two-step estimation with cluster-robust and Windmeijer (2005) finite-sample
  corrected standard errors.
- Forward orthogonal deviations (`transform=:fod`, Arellano-Bover) for Difference
  GMM, as an alternative to first differencing (better for unbalanced panels).
- `exog` keyword: strictly exogenous regressors are used as their own ("IV-style")
  GMM instruments (Roodman, 2009). A lag of a strictly exogenous covariate may
  itself be exogenous; the dependent variable and its lags may not.
- `time_effects` keyword: adds `T-2` exogenous period dummies to absorb common
  time shocks.
- `xtabond2`-style instrument lag limits (`min_lag`, `max_lag`) and automatic
  dropping of linearly dependent instrument columns (`drop_collinear`, on by default).
- Richer formula syntax: whitelisted transforms (`log`, `log10`, `log2`, `exp`,
  `sqrt`, `abs`) on either side and nested in lags (`log(y) ~ lag(log(y)) + log(x)`);
  explicit and ranged lags (`lag(x, 2)`, `lag(x, 0:1)`).
- StatsModels `@formula` support (`formula = @formula(y ~ lag(y) + x)`) in addition
  to the string form, plus the `lag` formula marker.
- [Tables.jl](https://github.com/JuliaData/Tables.jl) input: `fit`/`get_diff_data`
  accept any Tables-compatible source, not only `DataFrame`.
- Diagnostics: Sargan/Hansen J-test, Arellano-Bond AR(1)/AR(2) tests (with cached
  results, so `ar_test(model, order)` works after `diagnose`), Wald test,
  Jarque-Bera test, pseudo-R², instrument-proliferation check, and
  `diff_hansen_test` (a Durbin-Wu-Hausman-style difference-in-Hansen test for a
  nested instrument subset).
- StatsAPI `RegressionModel` interface (`coef`, `vcov`, `stderror`, `confint`, …),
  Stata-like result printing, and diagnostic plot recipes.
- Aqua.jl quality tests; ~99% line coverage.
- Package infrastructure: GitHub Actions CI (+ CompatHelper, TagBot), Documenter.jl
  docs, `.JuliaFormatter.toml`, `CHANGELOG.md`, `CONTRIBUTING.md`, and integration
  tests against a simulated panel with known parameters.

[Unreleased]: https://github.com/MichalS16/DynamicPanelModels/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/MichalS16/DynamicPanelModels/releases/tag/v0.3.0
