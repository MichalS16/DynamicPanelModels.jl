# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.2] — 2026-07-15

### Added

- Regression tests for three previously-uncovered defensive branches, found via
  line-coverage inspection rather than code reading alone: `build_instruments`'s
  abstract-model fallback (must raise an informative error, not a generic
  `MethodError`), `ar_test`'s degenerate `total_variance <= 0` guard (all-zero
  residuals), and `wald_test`'s near-singular `R*V*R'` regularization branch.

### Fixed

- `estimate` now rejects collinear regressors (`rank(X) < n_regressors`) with a
  clear error naming the affected columns, instead of failing deep inside the
  GMM solver with an opaque `LinearAlgebra.SingularException`. Found via
  adversarial testing (e.g. a regressor and an exact rescaling of it), not
  previously covered by any test.
- `examples/example.jl`'s diagnostics comment mischaracterized the AR(1)/AR(2)
  results: AR(1) is *expected* to be significant after differencing (not a
  misspecification signal, per the Getting Started guide); only AR(2)
  significance would indicate misspecification. Reworded for precision after
  running the example against live data and checking the actual output.

### Performance

- `calculate_clustered_weight_matrix` and `_ab_h_weight_matrix`'s per-cluster
  accumulation loops used non-mutating `A += ...`, reallocating the full
  `n_instruments × n_instruments` matrix on every individual/cluster instead of
  accumulating in place; changed to `A .+= ...` (identical result, no
  behavior change). Benchmarked on simulated panels: roughly 20-28% less
  memory allocated by `fit` at N=20,000-50,000 (e.g. two-step `DifferenceGMM`
  at N=50,000, T=10: ~4.08 GB → ~2.97 GB), confirming the README's large-N
  performance claim under an actual adversarial/stress-test pass rather than
  just reading the sparse-matrix code.

## [0.3.1] — 2026-07-14

### Added

- `lag` documented in `docs/src/api.md` (it was exported but missing from the
  API reference).
- Behavioral regression tests: `collapse`/`max_lags` are now verified to
  actually change `fit`'s returned coefficients/vcov (not just the
  intermediate instrument-matrix column count), and `SystemGMM(windmeijer=false)`
  is now verified to skip the correction the same way the existing
  `DifferenceGMM` case does.

### Changed

- `DifferenceGMM`/`SystemGMM` are now generated from a shared
  `@dynpanel_spec` macro to remove duplicated boilerplate between the two
  model spec structs; fields and behavior are unchanged.

### Fixed

- `DifferenceGMM`'s one-step weighting matrix used plain `(Z'Z)^-1` instead of
  the Arellano-Bond (1991) `(Z'HZ)^-1` (`H` reflecting the MA(1) structure of
  differenced homoskedastic errors). Coefficients were only mildly affected,
  but the one-step Sargan/Hansen J-statistic was biased downward by roughly a
  factor of the number of groups. Validated against the EmplUK dataset from
  the original paper (Table 4): one-step Sargan chi2(25) now ≈70.9 vs. the
  paper's 65.8, versus ≈0.5 before the fix.
- The new `DifferenceGMM` weighting-matrix code re-densified the sparse
  instrument matrix once per individual instead of once upfront, and called
  `Matrix()` twice on the same slice; fixed to match the existing
  densify-once pattern already used elsewhere in the solver. The
  cluster-boundary-walking loop this needed was also duplicated three times
  across the file; extracted into a shared, memoized `_cluster_ranges` helper
  (results threaded via a `ranges` keyword so callers don't recompute them).
- `calculate_clustered_weight_matrix`/`calculate_windmeijer_correction` now
  guard that an explicitly passed `ranges` keyword argument is actually
  consumed, instead of silently ignoring it.
- Added a genuine (non-tautological) Windmeijer correction regression test
  against a hand-computed expected value, replacing a check that only
  verified internal wiring.
- `_lag_vector`'s `(id, time) => row` index map was rebuilt from scratch for
  every regressor column in `ar_test` (once per column, all with `Any`-typed
  Dict keys causing boxing on every insert/lookup); it is now built once per
  `ar_test` call, concretely typed on `eltype(id)`/`eltype(time)`, and reused
  across all columns.
- `_solve_gmm` densified the instrument matrix `Z` unconditionally, even on
  the one-step, non-robust path where the densified copy is never used
  (`calculate_clustered_weight_matrix` is only called when `steps == 2` or
  `robust`); it is now only densified when actually needed.

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

[Unreleased]: https://github.com/MichalS16/DynamicPanelModels.jl/compare/v0.3.2...HEAD
[0.3.2]: https://github.com/MichalS16/DynamicPanelModels.jl/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/MichalS16/DynamicPanelModels.jl/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/MichalS16/DynamicPanelModels.jl/releases/tag/v0.3.0
