# src/diagnostics.jl

"""
    sargan_test(model::DynamicPanelResult) -> DynamicPanelTest

Compute the Hansen/Sargan J-test for overidentifying restrictions in a dynamic panel model.

# Arguments
- `model::DynamicPanelResult`: The fitted dynamic panel model containing coefficient estimates and test statistics.

# Returns
- `DynamicPanelTest`: A struct with the test name, J-statistic, degrees of freedom, and p-value.
"""
function sargan_test(model::DynamicPanelResult)
    # Degrees of freedom
    n_instruments = model.n_instruments
    n_regressors = length(model.coef)
    df = n_instruments - n_regressors

    # Handle edge case: non-positive degrees of freedom
    if df <= 0
        return DynamicPanelTest("Sargan J-Test", 0.0, 0, 1.0)
    end

    return DynamicPanelTest("Sargan J-Test", model.j_stat, df, model.j_pval)
end

"""
    diff_hansen_test(restricted::DynamicPanelResult, unrestricted::DynamicPanelResult) -> DynamicPanelTest

Difference-in-Hansen (C statistic) test for the validity of a subset of instruments,
following standard GMM dynamic panel practice (Roodman, 2009) — the panel-data
analogue of a Durbin-Wu-Hausman test for GMM instrument subsets.

`restricted` and `unrestricted` must be fit with the **same estimator** (both
`DifferenceGMM` or both `SystemGMM`) on the **same transformed sample** (same
`y`/`X`/`n_obs`) — e.g. the same model fit with and without additional `exog`
instruments, or with different `max_lags`/`collapse` settings. `unrestricted`
must use a strictly larger, nested instrument set. This test is **not** valid
for comparing `DifferenceGMM` against `SystemGMM`: System GMM adds level
equations and therefore changes the estimating equations and sample, not just
the instrument set, so the two are not nested in the required sense.

Under `H0` that the extra instruments used only in `unrestricted` are valid
(exogenous), the test statistic

    C = J_restricted - J_unrestricted

is asymptotically `χ²` distributed with degrees of freedom equal to the
difference in the number of overidentifying restrictions between the two
models. A large, significant `C` statistic indicates that the additional
instruments in `unrestricted` are not valid.

Because `J_restricted` and `J_unrestricted` are each computed with their own
independently-estimated two-step weighting matrix (rather than a single common
one), `C` is only guaranteed non-negative asymptotically; in finite samples —
especially with many instruments — it can come out slightly negative. This is
a well-documented artifact of the two-step weighting matrix, not a sign of
error (Roodman, 2009); such cases are reported as strong non-rejection of
`H0` (`C` clipped to `0`, `p-value = 1.0`) rather than as `NaN`.

# Arguments
- `restricted::DynamicPanelResult`: Model estimated with the smaller instrument set.
- `unrestricted::DynamicPanelResult`: Model estimated with the larger (nested) instrument set.

# Returns
- `DynamicPanelTest`: Test name `"Difference-in-Hansen (C)"`, the `C` statistic,
  its degrees of freedom, and p-value.

# Errors
Throws an error if the two models were not fit on the same sample (`n_obs`
differs) or if `unrestricted` does not have strictly more instruments than
`restricted`.
"""
function diff_hansen_test(restricted::DynamicPanelResult, unrestricted::DynamicPanelResult)
    if restricted.n_obs != unrestricted.n_obs
        error(
            "diff_hansen_test requires both models to be fit on the same sample " *
            "(same estimator and transformed data); got n_obs=$(restricted.n_obs) " *
            "vs n_obs=$(unrestricted.n_obs). Comparing DifferenceGMM against " *
            "SystemGMM is not a valid use of this test.",
        )
    end
    if unrestricted.n_instruments <= restricted.n_instruments
        error(
            "diff_hansen_test requires 'unrestricted' to have strictly more " *
            "instruments than 'restricted' (nested instrument sets); got " *
            "$(unrestricted.n_instruments) <= $(restricted.n_instruments).",
        )
    end

    df_r = restricted.n_instruments - length(restricted.coef)
    df_u = unrestricted.n_instruments - length(unrestricted.coef)
    df = df_u - df_r

    c_stat = max(restricted.j_stat - unrestricted.j_stat, 0.0)
    if df <= 0
        return DynamicPanelTest("Difference-in-Hansen (C)", 0.0, 0, 1.0)
    end
    pvalue = 1.0 - cdf(Chisq(df), c_stat)
    return DynamicPanelTest("Difference-in-Hansen (C)", c_stat, df, pvalue)
end

"""
    ar_test(model::DynamicPanelResult, order::Int, id::Vector, time::Vector)

Perform the Arellano–Bond test for serial correlation of order `order` (AR(order)).

# Arguments
- `model::DynamicPanelResult`: Fitted dynamic panel model containing residuals, regressors, and variance-covariance matrix.
- `order::Int`: Order `m` of the autocorrelation test.
- `id::Vector`: Vector of panel identifiers.
- `time::Vector`: Vector of time indices corresponding to each observation.

# Returns
- `DynamicPanelTest`: Test result including the test name, z-statistic, and p-value.
"""
function ar_test(model::DynamicPanelResult, order::Int, id::Vector, time::Vector)
    # Input validation
    if length(id) != model.n_obs || length(time) != model.n_obs
        error("Length of 'id' and 'time' must match the number of observations in the model.")
    end

    # Extract necessary components
    res = model.residuals
    X = model.X
    Z = model.Z
    V = model.vcov

    # Compute test statistic
    idx_map = _id_time_index(id, time)
    res_lag = _lag_vector(res, id, time, order, idx_map)
    numerator = dot(res, res_lag)
    var_moments = sum((res .* res_lag) .^ 2)

    # Compute variance of the estimation part (reuse idx_map across columns
    # instead of rebuilding it once per regressor)
    X_lag = similar(X)
    for i in axes(X, 2)
        X_lag[:, i] = _lag_vector(X[:, i], id, time, order, idx_map)
    end

    # Variance calculation
    d = - (X' * res_lag + X_lag' * res)
    var_estimation = dot(d, V * d)
    total_variance = var_moments + var_estimation

    # Handle edge case: non-positive variance
    if total_variance <= 0
        return DynamicPanelTest("Arellano-Bond AR($order)", NaN, 0, NaN)
    end

    # Compute z-statistic and p-value
    z_stat = numerator / sqrt(total_variance)
    pvalue = 2.0 * (1.0 - cdf(Normal(), abs(z_stat)))

    test_result = DynamicPanelTest("Arellano-Bond AR($order)", z_stat, 0, pvalue)
    ar_tests = get!(model.metadata, :ar_tests, Dict{Int,DynamicPanelTest}())
    ar_tests[order] = test_result

    return test_result
end

"""
    wald_test(model::DynamicPanelResult, R::Matrix{Float64}, r::Vector{Float64})

Performs a Wald test on the coefficients of a `DynamicPanelResult` model.

# Arguments
- `model::DynamicPanelResult`: The estimated dynamic panel model containing coefficients and variance-covariance matrix.
- `R::Matrix{Float64}`: Restriction matrix specifying linear constraints on the coefficients.
- `r::Vector{Float64}`: Vector of hypothesized values corresponding to the restrictions.

# Returns
- `DynamicPanelTest`: Struct containing the test name, Wald statistic, degrees of freedom, and p-value.
"""
function wald_test(model::DynamicPanelResult, R::Matrix{Float64}, r::Vector{Float64})
    # Extract coefficients and variance-covariance matrix
    β = model.coef
    V = model.vcov

    # Dimension checks
    if size(R, 2) != length(β)
        error("Dimension mismatch: R must have $(length(β)) columns.")
    end

    # Compute Wald statistic
    diff = R * β - r
    middle = R * V * R'

    # Regularization for numerical stability
    if cond(middle) > 1e12
        middle += 1e-9 * I
    end

    # Compute Wald statistic
    wald_stat = dot(diff, inv(middle) * diff)
    df = size(R, 1)
    pvalue = 1.0 - cdf(Chisq(df), wald_stat)

    return DynamicPanelTest("Wald Test", wald_stat, df, pvalue)
end

"""
    goodness_of_fit(model::DynamicPanelResult) -> Float64

Calculates the goodness-of-fit (R²) for a `DynamicPanelResult` model.

# Arguments
- `model::DynamicPanelResult`: The dynamic panel model result object containing observed and fitted values.

# Returns
- `Float64`: The R² value representing the proportion of variance explained by the model.
"""
function goodness_of_fit(model::DynamicPanelResult)
    return StatsAPI.r2(model)
end

"""
    check_proliferation(model::DynamicPanelResult) -> String

Checks for potential instrument proliferation in a dynamic panel model.

# Arguments
- `model::DynamicPanelResult`: The fitted dynamic panel model containing the number of instruments (`n_instruments`) and groups (`n_groups`).

# Returns
- `String`: A message indicating whether the number of instruments is acceptable or exceeds the number of groups, with a warning if proliferation may bias results.
"""
function check_proliferation(model::DynamicPanelResult)
    # Compute ratio
    ratio = model.n_instruments / model.n_groups
    msg = "Instruments: $(model.n_instruments), Groups: $(model.n_groups) (Ratio: $(round(ratio, digits=2)))"

    # Warning if instruments exceed groups
    if model.n_instruments > model.n_groups
        return "WARNING: " *
               msg *
               "\n -> Number of instruments exceeds groups. Results might be biased."
    end

    return "OK: " * msg
end

"""
    jarque_bera_test(model::DynamicPanelResult)

Performs the Jarque-Bera test for normality on the residuals of a `DynamicPanelResult` model.

# Arguments
- `model::DynamicPanelResult`: The fitted dynamic panel model whose residuals are tested.

# Returns
- `DynamicPanelTest`: A structure containing the test name, test statistic, degrees of freedom, and p-value.
"""
function jarque_bera_test(model::DynamicPanelResult)
    # Extract residuals
    e = model.residuals
    n = length(e)
    m = mean(e)
    s2 = mean((e .- m) .^ 2)
    m3 = mean((e .- m) .^ 3)
    m4 = mean((e .- m) .^ 4)
    skewness = m3 / s2^1.5
    kurtosis = m4 / s2^2
    jb_stat = (n / 6.0) * (skewness^2 + 0.25 * (kurtosis - 3.0)^2)
    pvalue = 1.0 - cdf(Chisq(2), jb_stat)

    return DynamicPanelTest("Jarque-Bera", jb_stat, 2, pvalue)
end

"""
Compute the lagged version of a vector within groups identified by `id`.

# Arguments
- `v::Vector`: The input values to lag.
- `id::Vector`: Group identifiers corresponding to each element in `v`.
- `time::Vector`: Time indices corresponding to each element in `v`.
- `k::Int`: The lag amount (number of time steps to shift).

# Returns
- `Vector`: A vector of the same length as `v` with lagged values; entries without a corresponding lag are zero.
"""
function _lag_vector(v::Vector, id::Vector, time::Vector, k::Int)
    idx_map = _id_time_index(id, time)
    return _lag_vector(v, id, time, k, idx_map)
end

"""
    _lag_vector(v, id, time, k, idx_map)

Same as `_lag_vector(v, id, time, k)`, but reuses a precomputed `(id, time) =>
row` index map (see [`_id_time_index`](@ref)) instead of rebuilding it —
callers lagging several columns against the same `(id, time)` (e.g.
`ar_test`'s per-regressor loop) should build the map once and pass it in.
"""
function _lag_vector(v::Vector, id::Vector, time::Vector, k::Int, idx_map::Dict)
    n = length(v)
    v_lagged = zeros(eltype(v), n)
    for i in 1:n
        target = (id[i], time[i] - k)
        if haskey(idx_map, target)
            v_lagged[i] = v[idx_map[target]]
        end
    end
    return v_lagged
end

"""
    _id_time_index(id::Vector, time::Vector)

Build a `(id, time) => row` index map for a panel, concretely typed on
`eltype(id)`/`eltype(time)` (rather than `Any`) so lookups avoid boxing.
"""
function _id_time_index(id::Vector, time::Vector)
    idx_map = Dict{Tuple{eltype(id),eltype(time)},Int}()
    for i in eachindex(id, time)
        idx_map[(id[i], time[i])] = i
    end
    return idx_map
end

"""
    diagnose(model::DynamicPanelResult; id=nothing, time=nothing)

Run GMM diagnostics on a `DynamicPanelResult` object and return a dictionary of results.

# Arguments
- `model::DynamicPanelResult`: The fitted dynamic panel model to diagnose.
- `id`: Optional vector identifying individual units for AR tests.
- `time`: Optional vector identifying time periods for AR tests.

# Returns
- `Dict{Symbol, Any}`: Contains results of Sargan test, AR tests (if `id` and `time` provided), 
  Jarque-Bera test, and pseudo R².
"""
function diagnose(model::DynamicPanelResult; id=nothing, time=nothing)
    # Use panel info from metadata if id/time not provided
    if isnothing(id) && haskey(model.metadata, :panel_info)
        id = [row.id for row in model.metadata[:panel_info]]
    end
    if isnothing(time) && haskey(model.metadata, :panel_info)
        time = [row.time for row in model.metadata[:panel_info]]
    end

    # Header
    println("=" ^ 60)
    println("      GMM DIAGNOSTIC REPORT")
    println("=" ^ 60)
    results = Dict{Symbol,Any}()

    # Sargan
    st = sargan_test(model)
    results[:sargan] = st
    @printf("Sargan J-test (validity):   stat = %8.3f, p-val = %6.4f\n", st.stat, st.pvalue)

    # AR Tests
    if !isnothing(id) && !isnothing(time)
        ar1 = ar_test(model, 1, id, time)
        ar2 = ar_test(model, 2, id, time)
        results[:ar1] = ar1
        results[:ar2] = ar2
        @printf("AR(1) test (serial corr):   stat = %8.3f, p-val = %6.4f\n", ar1.stat, ar1.pvalue)
        @printf("AR(2) test (serial corr):   stat = %8.3f, p-val = %6.4f\n", ar2.stat, ar2.pvalue)
    else
        println("AR tests skipped (provide 'id' and 'time' vectors).")
    end

    # Jarque-Bera
    jb = jarque_bera_test(model)
    results[:jarque_bera] = jb
    @printf("Jarque-Bera (normality):    stat = %8.3f, p-val = %6.4f\n", jb.stat, jb.pvalue)

    # Pseudo R2
    r2 = goodness_of_fit(model)
    results[:pseudo_r2] = r2
    println("-" ^ 60)
    println(check_proliferation(model))
    println("=" ^ 60)

    return results
end
