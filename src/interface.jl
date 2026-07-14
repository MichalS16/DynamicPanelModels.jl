# src/interface.jl

# libraries

"""
    StatsAPI.coef(model::DynamicPanelResult)

Return the estimated coefficients of a dynamic panel GMM model.

# Arguments
- `model::DynamicPanelResult`: A fitted dynamic panel GMM model.

# Returns
- `Vector{Float64}`: Coefficient estimates for each regressor.
"""
StatsAPI.coef(model::DynamicPanelResult) = model.coef

"""
    StatsAPI.vcov(model::DynamicPanelResult)

Return the variance–covariance matrix of the estimated coefficients
from a dynamic panel GMM model.

# Arguments
- `model::DynamicPanelResult`: A fitted dynamic panel GMM model.

# Returns
- `AbstractMatrix{T}`: Variance–covariance matrix of the estimated coefficients.
"""
StatsAPI.vcov(model::DynamicPanelResult) = model.vcov

"""
    StatsAPI.residuals(model::DynamicPanelResult)

Return the residuals of a dynamic panel GMM model.

# Arguments
- `model::DynamicPanelResult`: The fitted model containing residuals.

# Returns
- `Vector{Float64}`: Residuals for each observation after model transformation.
"""
StatsAPI.residuals(model::DynamicPanelResult) = model.residuals

"""
    StatsAPI.fitted(model::DynamicPanelResult)

Return the fitted values from a dynamic panel GMM model.

# Arguments
- `model::DynamicPanelResult`: A fitted dynamic panel model.

# Returns
- `Vector{Float64}`: Predicted values for each observation.
"""
StatsAPI.fitted(model::DynamicPanelResult) = model.fitted

"""
    StatsAPI.nobs(model::DynamicPanelResult; count_groups=false)

Return the number of observations or groups in a dynamic panel GMM model.

# Arguments
- `model::DynamicPanelResult`: Model result containing observations and groups.
- `count_groups::Bool=false`: If `true`, returns number of groups; otherwise returns total observations.

# Returns
- `Int`: Number of observations or groups.
"""
function StatsAPI.nobs(model::DynamicPanelResult; count_groups=false)
    return count_groups ? model.n_groups : model.n_obs
end

"""
    StatsAPI.coefnames(model::DynamicPanelResult)

Return the names of estimated coefficients for a dynamic panel GMM model.

# Arguments
- `model::DynamicPanelResult`: Model result containing coefficient metadata.

# Returns
- `Vector{String}`: Names of the estimated coefficients.
"""
StatsAPI.coefnames(model::DynamicPanelResult) = model.coef_names

"""
    StatsAPI.predict(model::DynamicPanelResult)

Return the predicted values from a dynamic panel GMM model.

# Arguments
- `model::DynamicPanelResult`: A fitted dynamic panel GMM model result.

# Returns
- `Vector{<:Real}`: Predicted (fitted) values for each observation.
"""
StatsAPI.predict(model::DynamicPanelResult) = model.fitted

"""
    StatsAPI.stderror(model::DynamicPanelResult)

Compute the standard errors of estimated coefficients from a dynamic panel GMM model,
i.e. `sqrt.(diag(vcov(model)))`. Whether these are homoskedastic, cluster-robust, or
Windmeijer-corrected depends on how the model was fit; see [`is_robust`](@ref) and
[`is_windmeijer`](@ref) to check which applies to a given result.

# Arguments
- `model::DynamicPanelResult`: Model result containing `vcov`.

# Returns
- `Vector{Float64}`: Standard errors.
"""
StatsAPI.stderror(model::DynamicPanelResult) = sqrt.(diag(vcov(model)))

"""
    StatsAPI.dof(model::DynamicPanelResult)

Return the degrees of freedom of a dynamic panel GMM model.

# Arguments
- `model::DynamicPanelResult`: Estimated model object.

# Returns
- `Int`: Number of estimated coefficients.
"""
StatsAPI.dof(model::DynamicPanelResult) = length(model.coef)

"""
    StatsAPI.dof_residual(model::DynamicPanelResult)

Return the residual degrees of freedom of a dynamic panel GMM model.

# Arguments
- `model::DynamicPanelResult`: Model result containing observations and estimated coefficients.

# Returns
- `Int`: Residual degrees of freedom (`n_obs - n_coef`).
"""
StatsAPI.dof_residual(model::DynamicPanelResult) = model.n_obs - length(model.coef)

"""
    formula(model::DynamicPanelResult)

Return the formula used for estimation from the model metadata.

`formula` is not part of StatsAPI (it belongs to StatsModels.jl); this package
defines and exports its own top-level `formula` function instead.

# Arguments
- `model::DynamicPanelResult`: Model result object.

# Returns
- `Any`: The stored formula from `metadata[:formula]`, or `"N/A"` if not available.
"""
formula(model::DynamicPanelResult) = get(model.metadata, :formula, "N/A")

"""
    StatsAPI.r2(model::DynamicPanelResult)

Compute the pseudo R² for a dynamic panel GMM model, defined as the squared 
correlation between observed and fitted values.

# Arguments
- `model::DynamicPanelResult`: Model result containing `y` and `fitted` values.

# Returns
- `Float64`: Pseudo R², or `NaN` if `fitted` is empty.
"""
function StatsAPI.r2(model::DynamicPanelResult)
    # Return NaN if fitted values are empty
    isempty(model.fitted) && return NaN

    return cor(model.y, model.fitted)^2
end

"""
    StatsAPI.confint(model::DynamicPanelResult; level::Real=0.95)

Compute two-sided confidence intervals for coefficients of a dynamic panel GMM model.

# Arguments
- `model::DynamicPanelResult`: Model result containing coefficients and standard errors.
- `level::Real=0.95`: Confidence level (default 0.95).

# Returns
- `Matrix{Float64}`: Two-column matrix with lower and upper bounds of the confidence intervals.
"""
function StatsAPI.confint(model::DynamicPanelResult; level::Real=0.95)
    # Calculate standard errors and z critical value
    se = stderror(model)
    z = quantile(Normal(), 1.0 - (1.0 - level) / 2.0)
    lower = model.coef .- z .* se
    upper = model.coef .+ z .* se

    return hcat(lower, upper)
end

"""
    StatsAPI.coeftable(model::DynamicPanelResult; level::Real=0.95)

Summarize the coefficients of a dynamic panel GMM model.

# Arguments
- `model::DynamicPanelResult`: Fitted dynamic panel model.
- `level::Real=0.95`: Confidence level for intervals (default 0.95).

# Returns
- `StatsAPI.CoefTable`: Table with columns `Estimate`, `Std. Error`, `z value`, 
  `Pr(>|z|)`, `Lower`, `Upper`, including significance stars for p-values.
"""
function StatsAPI.coeftable(model::DynamicPanelResult; level::Real=0.95)
    # Coefficients and z scores
    b = coef(model)
    se = stderror(model)
    z_score = b ./ se

    # P-values and confidence intervals
    p_values = 2.0 .* (1.0 .- cdf.(Normal(), abs.(z_score)))
    ci = confint(model; level=level)

    return CoefTable(
        hcat(b, se, z_score, p_values, ci[:, 1], ci[:, 2]),
        ["Estimate", "Std. Error", "z value", "Pr(>|z|)", "Lower $(level)", "Upper $(level)"],
        coefnames(model),
        4,
    )
end

"""
    StatsAPI.fit(model_spec::AbstractDynamicPanelModel, data;
                 formula, id_col::Symbol, time_col::Symbol, exog=String[], kwargs...)

Fit a dynamic panel model specified by `model_spec` using `data`.

The data are transformed according to `formula`, individual identifier `id_col`,
and time identifier `time_col`, then the model is estimated via `estimate`.

# Arguments
- `model_spec`: Dynamic panel model specification.
- `data`: Any [Tables.jl](https://github.com/JuliaData/Tables.jl)-compatible table
  (e.g. a `DataFrame`, a named tuple of columns, `CSV.File`, …).

# Keyword Arguments
- `formula`: Model formula, either a string (`"y ~ lag(y) + x"`) or a StatsModels
  `@formula` (`@formula(y ~ lag(y) + x)`). Supports `lag(var[, k|a:b])` for lags and
  the transforms `log, log10, log2, exp, sqrt, abs`. See [`get_diff_data`](@ref).
- `id_col`: Column identifying individuals.
- `time_col`: Column identifying time periods.
- `exog`: Names of right-hand-side regressors that are strictly exogenous; these are
  additionally used as their own ("IV-style") GMM instruments. See [`get_diff_data`](@ref).
- `time_effects`: If `true`, include period dummies (exogenous). See [`get_diff_data`](@ref).
- `transform`: `:fd` (first differences, default) or `:fod` (forward orthogonal
  deviations, Arellano-Bover; DifferenceGMM only). See [`get_diff_data`](@ref).
- `kwargs...`: Additional keyword arguments passed to `estimate` — including `steps`,
  `robust`, `collapse`, `max_lags`, `min_lag`, `max_lag`, and `drop_collinear`.

# Examples
```julia
fit(DifferenceGMM(robust=true), df; formula = @formula(y ~ lag(y) + x),
    id_col = :id, time_col = :time, exog = ["x"])
```
"""
function StatsAPI.fit(
    model_spec::AbstractDynamicPanelModel,
    data;
    formula::Union{AbstractString,FormulaTerm},
    id_col::Symbol,
    time_col::Symbol,
    exog::AbstractVector{<:AbstractString}=String[],
    time_effects::Bool=false,
    transform::Symbol=:fd,
    kwargs...,
)
    # Prepare differenced data (accepts any Tables.jl source and a string or @formula)
    diff_data = get_diff_data(
        data,
        id_col,
        time_col,
        formula,
        model_spec;
        exog=exog,
        time_effects=time_effects,
        transform=transform,
    )

    return estimate(model_spec, diff_data; kwargs...)
end

"""
    ngroups(model::DynamicPanelResult)

Return the number of unique cross-sectional groups in a dynamic panel GMM model.

# Arguments
- `model::DynamicPanelResult`: Model result containing group information.

# Returns
- `Int`: Number of groups in the model.
"""
ngroups(model::DynamicPanelResult) = model.n_groups

"""
    ninstruments(model::DynamicPanelResult)

Return the number of instruments used in a dynamic panel GMM model.

# Arguments
- `model::DynamicPanelResult`: Model result containing instrument information.

# Returns
- `Int`: Number of instruments used in the estimation.
"""
ninstruments(model::DynamicPanelResult) = model.n_instruments

"""
    is_robust(model::DynamicPanelResult)

Check if the dynamic panel GMM estimation used robust (cluster-sandwich) standard
errors, i.e. whether `robust=true` was in effect — this is `true` for one-step
robust fits as well as two-step fits, regardless of whether the Windmeijer
correction (see [`is_windmeijer`](@ref)) was additionally applied.

# Arguments
- `model::DynamicPanelResult`: The estimation result object.

# Returns
- `Bool`: `true` if robust standard errors were used, `false` for homoskedastic/naive SEs.
"""
is_robust(model::DynamicPanelResult) = model.robust

"""
    is_windmeijer(model::DynamicPanelResult)

Check if the Windmeijer (2005) finite-sample correction was applied to the
reported standard errors (only possible for two-step robust estimation).

# Arguments
- `model::DynamicPanelResult`: The estimation result object.

# Returns
- `Bool`: `true` if Windmeijer-corrected standard errors were applied, `false` otherwise.
"""
is_windmeijer(model::DynamicPanelResult) = model.windmeijer

"""
    ar_test(model::DynamicPanelResult, order::Int)

Retrieve a previously computed Arellano–Bond autocorrelation test of the given
order for a dynamic panel GMM model. AR tests are cached in `model.metadata[:ar_tests]`
the first time `diagnose(model; id, time)` or `ar_test(model, order, id, time)` is run.

# Arguments
- `model::DynamicPanelResult`: Model result containing AR test information.
- `order::Int`: Autocorrelation order to test (e.g., 1 for AR(1), 2 for AR(2)).

# Returns
- `DynamicPanelTest`: The cached AR(`order`) test result.

# Errors
Throws an error if the AR test for `order` has not yet been computed for this model.
"""
function ar_test(model::DynamicPanelResult, order::Int)
    ar_tests = get(model.metadata, :ar_tests, Dict{Int,DynamicPanelTest}())
    if !haskey(ar_tests, order)
        error(
            "AR($order) test not yet computed. Run diagnose(model; id, time) or " *
            "ar_test(model, $order, id, time) first.",
        )
    end
    return ar_tests[order]
end
