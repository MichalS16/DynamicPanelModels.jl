```@meta
CurrentModule = DynamicPanelModels
```

# API Reference

```@index
```

## Model Types

```@docs
DifferenceGMM
SystemGMM
AndersonHsiao
AbstractDynamicPanelModel
DynamicPanelResult
DynamicPanelTest
```

## Estimation and Data Handling

```@docs
fit
estimate
get_diff_data
parse_formula
build_instruments
lag
ngroups
ninstruments
is_robust
is_windmeijer
```

## StatsAPI Interface

```@docs
coef
vcov
stderror
residuals
fitted
predict
nobs
dof
dof_residual
confint
coeftable
formula
```

## Diagnostics and Tests

```@docs
sargan_test
ar_test
wald_test
goodness_of_fit
check_proliferation
jarque_bera_test
diagnose
diff_hansen_test
```
