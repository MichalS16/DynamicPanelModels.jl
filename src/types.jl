# src/types.jl

"""
    AbstractDynamicPanelModel

Abstract supertype representing dynamic panel data model specifications.
"""
abstract type AbstractDynamicPanelModel end

"""
    DifferenceGMM(; robust=false, steps=1, windmeijer=true)

Hyperparameters for the Difference GMM estimator of dynamic panel data models
(Arellano & Bond, 1991).

# Arguments
- `robust::Bool=false`: Compute robust standard errors if `true`.
- `steps::Int=1`: Number of GMM steps (`1` = one-step, `2` = two-step). Must be ≥ 1.
- `windmeijer::Bool=true`: Apply Windmeijer finite-sample correction when `robust=true`.
"""
struct DifferenceGMM <: AbstractDynamicPanelModel
    # init
    robust::Bool
    steps::Int
    windmeijer::Bool

    # constructor
    function DifferenceGMM(; robust::Bool=false, steps::Int=1, windmeijer::Bool=true)
        steps < 1 && throw(ArgumentError("steps must be >= 1"))
        return new(robust, steps, windmeijer)
    end
end

"""
    SystemGMM(; robust=false, steps=1, windmeijer=true)

Create a `SystemGMM` object representing hyperparameters for the
System GMM estimator of dynamic panel data models (Blundell & Bond, 1998).

# Arguments
- `robust::Bool=false`: Compute robust standard errors if `true`.
- `steps::Int=1`: Number of GMM steps (`1` = one-step, `2` = two-step). Must be `≥ 1`.
- `windmeijer::Bool=true`: Apply Windmeijer finite-sample correction for robust standard errors.
"""
struct SystemGMM <: AbstractDynamicPanelModel
    # init
    robust::Bool
    steps::Int
    windmeijer::Bool

    # constructor
    function SystemGMM(; robust::Bool=false, steps::Int=1, windmeijer::Bool=true)
        steps < 1 && throw(ArgumentError("steps must be >= 1"))
        return new(robust, steps, windmeijer)
    end
end

"""
    AndersonHsiao()

Anderson–Hsiao (1981) instrumental-variables estimator for dynamic panel data models.
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
