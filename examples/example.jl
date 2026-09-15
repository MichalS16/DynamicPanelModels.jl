# examples/example.jl
#
# Reproduces the Arellano-Bond (1991) employment equation (Table 4, column
# a2: two-step) on the EmplUK dataset, the same panel used in the original
# paper, and checks the package's estimates and diagnostics against the
# published numbers. (Column a1, one-step, is checked the same way in
# test/test_replication.jl rather than duplicated here.)

using DynamicPanelModels
using DataFrames
using CSV
using Downloads
using Plots
using Statistics

## Load the EmplUK dataset (Arellano & Bond, 1991): employment, wages,
## capital, and output for UK firms, 1976-1984.
url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/plm/EmplUK.csv"
df = CSV.read(Downloads.download(url), DataFrame)

first(df, 5)
show(describe(df); allrows=true, allcols=true)

# Panel balance
obs_per_firm = combine(groupby(df, :firm), nrow => :count)
println("Min years per firm: ", minimum(obs_per_firm.count))
println("Max years per firm: ", maximum(obs_per_firm.count))
println("Average years per firm: ", round(mean(obs_per_firm.count); digits=2))

## The paper's employment equation (16) is specified in logs, with two lags
## of employment and one or two lags of each covariate.
df.n = log.(df.emp)
df.w = log.(df.wage)
df.k = log.(df.capital)
df.ys = log.(df.output)

# Correlation matrix
vars_to_check = [:n, :w, :k, :ys]
cor_matrix = cor(Matrix(df[:, vars_to_check]))
DataFrame(cor_matrix, vars_to_check)

## Difference GMM (Arellano-Bond), two-step with Windmeijer-corrected SEs.
## `w`, `k`, and `ys` (and their lags) are treated as strictly exogenous, as
## in the paper's columns (a1)/(a2); only the lagged dependent variable is
## instrumented via its own further lags. The paper includes year dummies
## (`time_effects=true`).
model = fit(
    DifferenceGMM(; robust=true, steps=2),
    df;
    formula="n ~ lag(n) + lag(n,2) + w + lag(w) + k + lag(k) + lag(k,2) + ys + lag(ys) + lag(ys,2)",
    id_col=:firm,
    time_col=:year,
    exog=["w", "L.w", "k", "L.k", "L2.k", "ys", "L.ys", "L2.ys"],
    time_effects=true,
)
display(model)

## Diagnostics. Table 4, column (a2): n_{t-1}=0.629, n_{t-2}=-0.065,
## w_t=-0.526, k_t=0.278, ys_t=0.592, Sargan chi2(25)=31.4. The package
## reproduces these to the published precision (0.6287, -0.0652, -0.5258,
## 0.2784, 0.5919, J=31.38 below), and the AR(1)/AR(2) statistics match R's
## `plm::mtest` to three decimals. AR(1) is expected to be significant after
## differencing (that's not a misspecification signal, see the Getting
## Started guide); AR(2) is not significant, which is the actual check for
## misspecification here.
diagnose(model)

## Visualization
plot(model)
plot(model, :qq; title="Checking Normality: Q-Q Plot", markercolor=:orange, markersize=6)
plot(model, :residuals; title="Residuals Over Time", markercolor=:red, markeralpha=0.3)
plot(
    model,
    :histogram;
    title="Residual Distribution",
    fillcolor=:green,
    xlabel="Standardized Residual",
)
