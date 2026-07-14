# examples/example.jl

# libraries
using DynamicPanelModels
using DataFrames
using CSV
using Downloads
using Plots
using Statistics
using LinearAlgebra

## Load example dataset (EmplUK - Arellano & Bond, 1991)
# Dataset contains data on employment, wages, capital, and output for UK firms from 1976 to 1984.
url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/plm/EmplUK.csv"
df = CSV.read(Downloads.download(url), DataFrame)

## Dataset exploration
# look at the first few rows
first(df, 5)

# Summary of the dataset
show(describe(df); allrows=true, allcols=true)

# Panel Balance Check
obs_per_firm = combine(groupby(df, :firm), nrow => :count)
println("Min years per firm: ", minimum(obs_per_firm.count))
println("Max years per firm: ", maximum(obs_per_firm.count))
println("Average years per firm: ", round(mean(obs_per_firm.count); digits=2))

# Correlation Matrix
vars_to_check = [:emp, :wage, :capital, :output]
cor_matrix = cor(Matrix(df[:, vars_to_check]))
DataFrame(cor_matrix, vars_to_check)

### Estimations
### Equation: emp ~ lag(emp) + wage + capital + output
### Note: wage/capital/output are treated as endogenous here (instrumented via
### their own lags, as in standard Arellano-Bond). If a regressor is strictly
### exogenous, pass its name via `exog=["varname"]` to `fit` to additionally use
### it as its own ("IV-style") instrument, e.g. `exog=["output"]`.
## Difference GMM (Arellano-Bond)
model_diff = fit(
    DifferenceGMM(; robust=true),
    df;
    formula="emp ~ lag(emp) + wage + capital + output",
    id_col=:firm,
    time_col=:year,
);
display(model_diff)

# Diagnostic
diagnose(model_diff)

## System GMM (Blundell-Bond)
model_sys = fit(
    SystemGMM(; robust=true),
    df;
    formula="emp ~ lag(emp) + wage + capital + output",
    id_col=:firm,
    time_col=:year,
);
display(model_sys)

# Diagnostic
diagnose(model_sys)

## Detailed diagnostic
# Arellano-Bond Autocorrelation Tests
# Extract id and time vectors from metadata
id_vec = [row.id for row in model_diff.metadata[:panel_info]];
time_vec = [row.time for row in model_diff.metadata[:panel_info]];
ar1_res = ar_test(model_diff, 1, id_vec, time_vec);
ar2_res = ar_test(model_diff, 2, id_vec, time_vec);
println("Test Name: ", ar2_res.test_name)
println("AR(2) Stat: ", round(ar2_res.stat; digits=3))
println("AR(2) p-val: ", round(ar2_res.pvalue; digits=4))

# Sargan J-Test for Overidentifying Restrictions
sar_res = sargan_test(model_diff);
println("J-statistic: ", round(sar_res.stat; digits=3))
println("Degrees of Freedom: ", sar_res.dof)
println("Sargan p-value: ", round(sar_res.pvalue; digits=4))

# Wald Test for Overall Significance
n_params = length(model_diff.coef);
R = Matrix{Float64}(I, n_params, n_params);
r = zeros(n_params);
wald_res = wald_test(model_diff, R, r);
println("Wald Stat: ", round(wald_res.stat; digits=3))
println("Wald p-val: ", round(wald_res.pvalue; digits=4))

# Goodness-of-Fit (Pseudo R²) & Proliferation Check
r2_val = goodness_of_fit(model_diff);
prolif_msg = check_proliferation(model_diff);
println("Pseudo R²: ", round(r2_val; digits=4))
println(prolif_msg)

# Residual Normality Test (Jarque-Bera)
jb_res = jarque_bera_test(model_diff);
println("JB Stat: ", round(jb_res.stat; digits=3))
println("JB p-val: ", round(jb_res.pvalue; digits=4))

## Visualization of results
# All plots for System GMM model
plot(model_sys)

# Q-Q Plot of Residuals
plot(model_sys, :qq; title="Checking Normality: Q-Q Plot", markercolor=:orange, markersize=6)

# Residuals vs Fitted Values
plot(model_sys, :residuals; title="Time Series of Errors", markercolor=:red, markeralpha=0.3)

# Histogram of Residuals
plot(
    model_sys,
    :histogram;
    title="Residual Distribution True (title)",
    fillcolor=:green,
    xlabel="Standardized Deviation (xlabel)",
)
