# src/types.jl

"""
    AbstractDynamicPanelModel

Abstract supertype representing dynamic panel data model specifications.
"""
abstract type AbstractDynamicPanelModel end

# Shared validation for the `steps` field of DifferenceGMM/SystemGMM.
_validate_steps(steps::Int) = steps < 1 && throw(ArgumentError("steps must be >= 1"))

# DifferenceGMM and SystemGMM carry identical robust/steps/windmeijer fields
# and constructor logic, but must stay distinct *types* (multiple dispatch on
# estimator type throughout estimation.jl/instruments.jl relies on it, e.g.
# `initial_weight_matrix(::DifferenceGMM, ...)` vs the generic fallback). This
# macro generates one struct + constructor from a name and docstring, so the
# two definitions can't drift apart, without merging them into one type.
macro dynpanel_spec(name, docstring)
    quote
        @doc $(esc(docstring)) struct $(esc(name)) <: AbstractDynamicPanelModel
            robust::Bool
            steps::Int
            windmeijer::Bool

            function $(esc(name))(; robust::Bool=false, steps::Int=1, windmeijer::Bool=true)
                _validate_steps(steps)
                return new(robust, steps, windmeijer)
            end
        end
    end
end

@dynpanel_spec DifferenceGMM """
    DifferenceGMM(; robust=false, steps=1, windmeijer=true)

Hyperparameters for the Difference GMM estimator of dynamic panel data models
(Arellano & Bond, 1991).

First-differencing eliminates the individual fixed effect `η_i` from
`y_it = α y_{i,t-1} + x_it'β + η_i + ε_it`, giving
`Δy_it = α Δy_{i,t-1} + Δx_it'β + Δε_it`. Lagged levels `y_{i,t-2}, y_{i,t-3}, …`
are valid instruments for `Δy_{i,t-1}` under the assumption that `ε_it` is not
serially correlated (so `E[Δε_it · y_{i,t-s}] = 0` for `s ≥ 2`); this is what
[`ar_test`](@ref) checks (a significant AR(2) statistic on the differenced
residuals signals misspecification). One-step uses the Arellano-Bond (1991,
p.279) weighting matrix `A_N = N⁻¹ Σᵢ Zᵢ'HZᵢ` (`H` reflecting the MA(1)
structure that differencing induces in homoskedastic errors); two-step
re-weights by the estimated one-step residual covariance and, when
`windmeijer=true`, applies the Windmeijer (2005) finite-sample correction to
avoid understating the two-step standard errors.

# Arguments
- `robust::Bool=false`: Compute robust standard errors if `true`.
- `steps::Int=1`: Number of GMM steps (`1` = one-step, `2` = two-step). Must be ≥ 1.
- `windmeijer::Bool=true`: Apply Windmeijer finite-sample correction when `robust=true`.
"""

@dynpanel_spec SystemGMM """
    SystemGMM(; robust=false, steps=1, windmeijer=true)

Create a `SystemGMM` object representing hyperparameters for the
System GMM estimator of dynamic panel data models (Blundell & Bond, 1998).

Stacks the Difference GMM equations with an additional levels equation
`y_it = α y_{i,t-1} + x_it'β + η_i + ε_it`, instrumented with *lagged
differences* `Δy_{i,t-1}, Δx_{i,t-1}` rather than lagged levels. This is valid
under the mean-stationarity assumption `E[η_i Δy_{i,t}] = 0` (deviations of
the initial condition from its long-run mean are uncorrelated with that
mean) — see Blundell & Bond (1998, Sec. 2.3.1). System GMM is most useful
when the autoregressive parameter is close to unity, where lagged levels
become weak instruments for `Δy_{i,t-1}` in plain Difference GMM. Because the
extra identifying assumption is not implied by the model and is not
automatically tested, inspect [`sargan_test`](@ref)/[`diff_hansen_test`](@ref)
before trusting System GMM estimates on highly persistent series.

# Arguments
- `robust::Bool=false`: Compute robust standard errors if `true`.
- `steps::Int=1`: Number of GMM steps (`1` = one-step, `2` = two-step). Must be `≥ 1`.
- `windmeijer::Bool=true`: Apply Windmeijer finite-sample correction for robust standard errors.
"""

"""
    AndersonHsiao()

Anderson–Hsiao (1981) instrumental-variables estimator for dynamic panel data models.

Estimates the first-differenced equation `Δy_it = α Δy_{i,t-1} + Δx_it'β + Δε_it`
by IV, instrumenting `Δy_{i,t-1}` with a single lagged level `y_{i,t-2}` (or,
equivalently, `Δy_{i,t-2}`). Unlike Difference/System GMM it uses exactly one
instrument for the lagged-difference regressor rather than the full set of
valid lags, so it is consistent but generally less efficient; it has no
tuning hyperparameters and serves as a simple baseline for comparison.
"""
struct AndersonHsiao <: AbstractDynamicPanelModel end

"""
    DynamicPanelResult <: RegressionModel

Result of dynamic panel data GMM estimation (e.g. Difference GMM, System GMM).

# Fields
- `coef::Vector{T}`: Estimated coefficients.
- `vcov::AbstractMatrix{T}`: Variance–covariance matrix.
- `residuals::Vector{T}`: Model residuals.
- `fitted::Vector{T}`: Fitted values.
- `n_obs::Int`: Number of observations.
- `n_groups::Int`: Number of cross-sectional units.
- `n_instruments::Int`: Number of instruments.
- `coef_names::Vector{String}`: Names of coefficients.
- `X::AbstractMatrix{T}`: Transformed regressor matrix.
- `y::Vector{T}`: Transformed response.
- `Z::AbstractSparseMatrix{T}`: Instrument matrix.
- `W::AbstractMatrix{T}`: GMM weighting matrix.
- `j_stat::T`: Hansen–Sargan J statistic.
- `j_pval::T`: P-value of the J test.
- `windmeijer::Bool`: Windmeijer correction applied (only meaningful when `robust`).
- `robust::Bool`: Robust (cluster-sandwich) standard errors were used.
- `metadata::Dict{Symbol,Any}`: Additional metadata.

# Constructors
    DynamicPanelResult(coef, vcov, residuals, fitted,
                       n_obs, n_groups, n_instruments, coef_names,
                       X, y, Z, W, j_stat, j_pval,
                       windmeijer, metadata; robust=windmeijer)

    DynamicPanelResult(coef, vcov, residuals, fitted,
                       n_obs, n_groups, n_instruments, coef_names,
                       X, y, Z, W, j_stat, j_pval;
                       windmeijer=false, robust=windmeijer, metadata=Dict())
"""
struct DynamicPanelResult{T<:Real,MT<:AbstractMatrix{T},ST<:AbstractSparseMatrix{T}} <:
       StatsAPI.RegressionModel
    # Estimated parameters
    coef::Vector{T}
    vcov::MT
    residuals::Vector{T}
    fitted::Vector{T}

    # Dimensions
    n_obs::Int
    n_groups::Int
    n_instruments::Int

    # Metadata names
    coef_names::Vector{String}

    # Model Matrices
    X::MT
    y::Vector{T}
    Z::ST
    W::MT

    # Diagnostic Statistics
    j_stat::T
    j_pval::T

    # Additional Metadata
    windmeijer::Bool
    robust::Bool
    metadata::Dict{Symbol,Any}

    # Inner constructor
    function DynamicPanelResult(
        coef::Vector{T},
        vcov::MT,
        residuals::Vector{T},
        fitted::Vector{T},
        n_obs::Int,
        n_groups::Int,
        n_inst::Int,
        coef_names::Vector{String},
        X::MT,
        y::Vector{T},
        Z::ST,
        W::MT,
        j_stat::T,
        j_pval::T,
        windmeijer::Bool,
        metadata::Dict{Symbol,Any};
        robust::Bool=windmeijer,
    ) where {T<:Real,MT<:AbstractMatrix{T},ST<:AbstractSparseMatrix{T}}

        # Dimension checks
        length(coef) != size(X, 2) && throw(DimensionMismatch("Coefficients/X mismatch"))
        return new{T,MT,ST}(
            coef,
            vcov,
            residuals,
            fitted,
            n_obs,
            n_groups,
            n_inst,
            coef_names,
            X,
            y,
            Z,
            W,
            j_stat,
            j_pval,
            windmeijer,
            robust,
            metadata,
        )
    end
end

# Convenience constructor
function DynamicPanelResult(
    coef::Vector{T},
    vcov::AbstractMatrix{T},
    residuals::Vector{T},
    fitted::Vector{T},
    n_obs::Int,
    n_groups::Int,
    n_inst::Int,
    names::Vector{String},
    X::AbstractMatrix{T},
    y::Vector{T},
    Z::AbstractSparseMatrix{T},
    W::AbstractMatrix{T},
    j_s::T,
    j_p::T;
    windmeijer::Bool=false,
    robust::Bool=windmeijer,
    metadata=Dict{Symbol,Any}(),
) where {T<:Real}
    return DynamicPanelResult(
        coef,
        vcov,
        residuals,
        fitted,
        n_obs,
        n_groups,
        n_inst,
        names,
        X,
        y,
        Z,
        W,
        j_s,
        j_p,
        windmeijer,
        metadata;
        robust=robust,
    )
end

"""
    DynamicPanelTest

Results of a diagnostic test for dynamic panel data models.

# Fields
- `test_name::String`: Test identifier.
- `stat::Float64`: Test statistic.
- `dof::Int`: Degrees of freedom.
- `pvalue::Float64`: Associated p-value.
"""
struct DynamicPanelTest
    test_name::String
    stat::Float64
    dof::Int
    pvalue::Float64
end
