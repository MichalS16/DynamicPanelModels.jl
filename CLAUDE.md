# CLAUDE.md — DynamicPanelModels.jl

Julia package for dynamic panel GMM estimation (Arellano-Bond Difference GMM,
Blundell-Bond System GMM, Anderson-Hsiao IV) with robust/Windmeijer SEs and
GMM diagnostics. School final project — **econometric correctness is the top
priority**; verify formulas against the source papers (Arellano-Bond 1991,
Blundell-Bond 1998, Windmeijer 2005, Roodman 2009), not just "tests pass".

## Commands

```bash
julia --project=. -e 'using Pkg; Pkg.test()'   # run full test suite (Aqua + unit + integration)
julia --project=docs docs/make.jl              # build Documenter.jl docs locally
# Format (blue style, margin 100 — config in .JuliaFormatter.toml):
julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.add("JuliaFormatter"); using JuliaFormatter; format("src"); format("test")'
```

Requires Julia 1.12+.

## Architecture (`src/`)

Estimation pipeline: `fit` → `get_diff_data` → `build_instruments` → `estimate` → `_solve_gmm`.

- `DynamicPanelModels.jl` — module entry point: `include`s all files + the `export` list. **Add every new public function's `export` here.**
- `types.jl` — `DifferenceGMM`/`SystemGMM`/`AndersonHsiao` specs, `DynamicPanelResult`, `DynamicPanelTest`.
- `transformations.jl` — `parse_formula`, `get_diff_data` (differencing, lags, `exog_idx`).
- `instruments.jl` — `build_instruments` per estimator; `_build_ab_instruments` + `append_exog_instruments` helpers.
- `estimation.jl` — GMM solver, clustered weight matrix, Windmeijer correction, `posdef_fix`.
- `diagnostics.jl` — Sargan, AR(m), Wald, Jarque-Bera, `diff_hansen_test`, `diagnose`.
- `interface.jl` — StatsAPI accessors (`coef`, `vcov`, `stderror`, …), `fit`.
- `show.jl` / `plot_recipes.jl` — pretty printing and RecipesBase plots.

Tests mirror `src/` one-to-one (`test/test_*.jl`) plus `test/test_integration.jl`
(simulated DGP with known params). Add new files to `test/runtests.jl`.

## Gotchas

- **`steps`/`robust` come from the model spec**, defaulted in `estimate` via
  `_model_steps`/`_model_robust`. `fit(DifferenceGMM(steps=2), …)` must actually
  do two-step — don't let `estimate`'s own kwargs silently override the spec.
- **`exog` kwarg**: strictly-exogenous regressors must be passed via
  `fit(…; exog=["x"])` to be used as their own IV-style instruments. Without it
  their coefficient is only weakly identified (biased). A lag of an exogenous
  covariate (`"L.x"`) may be `exog`; the dependent variable / its lag may not.
- **Formula syntax** (`transformations.jl`): `lag(v[, k|a:b])` + whitelisted
  transforms (`log/log10/log2/exp/sqrt/abs`), nestable (`lag(log(x))`). Parsed by
  `_parse_term`; add new transforms to the `_TRANSFORMS` whitelist (no `eval`).
- **`time_effects=true`** adds only `T-2` period dummies (drop first two); using
  `T-1` makes the differenced regressor matrix rank-deficient → variance blow-up.
- **`transform=:fod`** (forward orthogonal deviations) shifts instrument timing:
  FOD instruments start at `y_{t-1}` (offset 0), FD at `y_{t-2}` (offset 1) — see
  the `offset` in `_build_ab_instruments`. `:fod` is DifferenceGMM-only.
- **Instrument control** (`estimation.jl`/`instruments.jl`): `min_lag`/`max_lag`
  bound lag *order*, `max_lags` caps the *count*; `drop_collinear=true` (default)
  removes rank-deficient instrument columns via pivoted QR (`_drop_collinear_columns`).
- **First difference starts at t=2** in `get_diff_data` (not t=3); dropping t=2
  wastes valid observations. Level-equation rows (System GMM) start at t=3.
- **Windmeijer correction** (`calculate_windmeijer_correction`): `D = ∂β2/∂β1`
  must preserve `Ω1`'s **per-cluster** structure, use `Z'û2` (not `Z'X·β2`), and
  assemble `V2 + D·V2 + V2·D' + D·V1r·D'` with the one-step **robust** `V1r`.
  Getting any of these wrong silently explodes the SEs — validate against EmplUK
  (lagged-term SE ≈ 0.17); regression guard: `test_integration.jl`
  "Windmeijer correction sanity".
- **New model-spec kwargs need behavioral tests, not constructor-only tests.**
  (`windmeijer=false` was dead code for a session because its only test checked
  the field, never `fit`'s actual `vcov`.) Test the numeric effect through
  `estimate`/`_solve_gmm`, not just that the field is stored.
- **`is_robust` vs `is_windmeijer`**: `is_robust(model)` = were SEs
  cluster-robust at all (true for one-step robust too); `is_windmeijer(model)` =
  was the two-step Windmeijer correction applied (`model.windmeijer` only means
  something when `steps==2`). Don't conflate them.
- **`diff_hansen_test`** is only valid for same-estimator, same-sample, nested
  instrument sets — NOT DifferenceGMM vs SystemGMM (different transformed data).
- `Manifest.toml` / `docs/build/` are gitignored; don't commit them.
- `.markdownlint.json` exists (MD013/MD024/MD033 relaxed for CHANGELOG's
  repeated `### Added` headings and HTML in badges) — intentional, not stray.
