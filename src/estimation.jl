# src/estimation.jl

"""
    estimate(model::AbstractDynamicPanelModel, diff_data::NamedTuple;
             steps::Int=model.steps, robust::Bool=model.robust,
             collapse::Bool=false, max_lags::Int=999)

Estimate a dynamic panel GMM model (Difference or System GMM).

By default, `steps` and `robust` are taken from the `model` specification
(e.g. `DifferenceGMM(robust=true, steps=2)`); passing them explicitly as
keyword arguments to `estimate`/`fit` overrides the model's settings.

# Arguments
- `model::AbstractDynamicPanelModel`: Either `DifferenceGMM` or `SystemGMM`.
- `diff_data::NamedTuple`: Preprocessed panel data with fields:
    - `y`: Response vector (differenced if needed)
    - `X`: Design matrix
    - `coef_names`: Names of regressors
    - `panel_info`: Row metadata (`id`, `time`, `is_level`)
    - `n_obs`: Number of observations
    - `id_time_to_y`: Dict mapping id => Dict(time => y)
    - `id_time_to_diff_y`: Dict mapping id => Dict(time => Δy)
    - `valid_times`: Vector of observed time periods
- Keyword arguments:
    - `steps::Int`: Number of GMM steps (1 or 2). Defaults to `model.steps`.
    - `robust::Bool`: Use robust (cluster-sandwich) standard errors. Defaults to `model.robust`.
    - `windmeijer::Bool`: When `steps == 2 && robust`, apply the Windmeijer (2005)
      finite-sample correction instead of the naive two-step sandwich. Defaults to
      `model.windmeijer`. Has no effect for one-step estimation.
    - `collapse::Bool=false`: Collapse instruments to one column per lag.
    - `max_lags::Int=999`: Maximum instrument lag count per period.
    - `min_lag::Int=1`, `max_lag::Int=typemax(Int)`: bounds on the instrument lag order.
    - `drop_collinear::Bool=true`: drop linearly dependent instrument columns (as `xtabond2`).

# Returns
- `DynamicPanelResult`: Contains coefficients, standard errors, residuals, fitted values, instruments, and diagnostics.
"""
function estimate(
    model::AbstractDynamicPanelModel,
    diff_data::NamedTuple;
    steps::Int=_model_steps(model),
    robust::Bool=_model_robust(model),
    windmeijer::Bool=_model_windmeijer(model),
    collapse::Bool=false,
    max_lags::Int=999,
    min_lag::Int=1,
    max_lag::Int=typemax(Int),
    drop_collinear::Bool=true,
)
    # Extract data
    y = diff_data.y
    X = diff_data.X
    coef_names = diff_data.coef_names
    n_obs, n_regressors = size(X)

    # Check sufficient observations
    if n_obs < n_regressors
        error("Insufficient observations ($n_obs) for $n_regressors regressors.")
    end

    # Build instruments
    Z = build_instruments(
        model, diff_data; collapse=collapse, max_lags=max_lags, min_lag=min_lag, max_lag=max_lag
    )
    # Drop linearly dependent instrument columns (as xtabond2 does automatically);
    # collinear instruments add nothing and destabilize the weighting matrix.
    if drop_collinear
        Z = _drop_collinear_columns(Z)
    end
    n_instruments = size(Z, 2)

    # Check identification
    if n_instruments < n_regressors
        error("Model under-identified. Instruments ($n_instruments) < Regressors ($n_regressors).")
    end

    # Solve GMM
    return _solve_gmm(y, X, Z, diff_data, coef_names, steps, robust, windmeijer, model)
end

# Defaults for `steps`/`robust`/`windmeijer` sourced from the model spec.
# DifferenceGMM and SystemGMM carry these fields; estimators without them
# (e.g. AndersonHsiao) fall back to `default` via the abstract-type method.
_model_field(model::Union{DifferenceGMM,SystemGMM}, field::Symbol, default) = getfield(model, field)
_model_field(::AbstractDynamicPanelModel, ::Symbol, default) = default

_model_steps(model::AbstractDynamicPanelModel) = _model_field(model, :steps, 1)
_model_robust(model::AbstractDynamicPanelModel) = _model_field(model, :robust, true)
_model_windmeijer(model::AbstractDynamicPanelModel) = _model_field(model, :windmeijer, true)

"""
    _drop_collinear_columns(Z; tol=1e-9) -> SparseMatrixCSC

Return `Z` with linearly dependent columns removed (keeping, from each collinear
set, the earliest column). Uses a rank-revealing QR with column pivoting; the
retained columns are reported in their original order. This mirrors the automatic
dropping of redundant instruments in Stata's `xtabond2`.
"""
function _drop_collinear_columns(Z::AbstractMatrix; tol::Real=1e-9)
    p = size(Z, 2)
    p <= 1 && return Z
    M = Matrix(Z)
    F = qr(M, ColumnNorm())
    d = abs.(diag(F.R))
    isempty(d) && return Z
    r = count(>(tol * maximum(d)), d)
    r == p && return Z                     # full rank: nothing to drop
    keep = sort(F.p[1:r])
    return sparse(M[:, keep])
end

"""
    _solve_gmm(y, X, Z, diff_data, coef_names, steps, robust, windmeijer, model_type)

Internal solver for dynamic panel GMM estimation.

# Arguments
- `y::AbstractVector`: Dependent variable vector.
- `X::AbstractMatrix`: Regressor matrix.
- `Z::AbstractMatrix`: Instrument matrix.
- `diff_data`: Data structure containing panel information and differenced variables.
- `coef_names::Vector{String}`: Names of the coefficients.
- `steps::Int`: Number of GMM steps (1 or 2).
- `robust::Bool`: Whether to compute robust (cluster-sandwich) standard errors.
- `windmeijer::Bool`: When `steps == 2 && robust`, apply the Windmeijer finite-sample
  correction; when `false`, use the naive (uncorrected) two-step sandwich instead.
- `model_type`: Type of GMM model (e.g., difference or system).

# Returns
- `DynamicPanelResult`: Struct containing estimated coefficients, variance-covariance matrix, residuals, fitted values, J-test statistics, and diagnostic information.
"""
function _solve_gmm(y, X, Z, diff_data, coef_names, steps, robust, windmeijer, model_type)
    # Precomputations of dimensions and moments
    n_obs, n_reg = size(X)
    ZtX = Z' * X
    Zty = Z' * y

    # Initial One-Step GMM Estimation
    W1 = initial_weight_matrix(model_type, Z, diff_data)
    bread1 = inv(posdef_fix(ZtX' * W1 * ZtX))
    β1 = bread1 * (ZtX' * W1 * Zty)
    res1 = y - X * β1
    β, W, final_bread = β1, W1, bread1
    vcov = Matrix{Float64}(undef, n_reg, n_reg)
    is_windmeijer = false

    # Two-Step GMM
    if steps == 2
        # Second-Step Estimation
        Ω_clustered = calculate_clustered_weight_matrix(
            Z, res1, diff_data.panel_info, diff_data.n_groups
        )
        W2 = inv(posdef_fix(Ω_clustered))
        bread2 = inv(posdef_fix(ZtX' * W2 * ZtX))
        β2 = bread2 * (ZtX' * W2 * Zty)
        β, W, final_bread = β2, W2, bread2

        # Variance-Covariance Matrix
        if robust && windmeijer
            # One-step robust (sandwich) covariance, used in the Windmeijer correction
            V1r = bread1 * (ZtX' * W1 * Ω_clustered * W1 * ZtX) * bread1
            # Windmeijer Finite-Sample Correction
            vcov = calculate_windmeijer_correction(
                Z, X, res1, β2, ZtX, Zty, W2, bread2, V1r, diff_data.panel_info
            )
            is_windmeijer = true
        else
            # Naive (uncorrected) two-step sandwich, whether or not `robust` was
            # requested — without the Windmeijer correction there is no separate
            # "robust" two-step variance to fall back to; `bread2` already comes
            # from the efficient two-step weighting matrix.
            vcov = bread2
        end

    else
        # One-Step Variance
        if robust
            # Clustered Robust SEs
            Ω_clustered = calculate_clustered_weight_matrix(
                Z, res1, diff_data.panel_info, diff_data.n_groups
            )
            vcov = bread1 * (ZtX' * W1 * Ω_clustered * W1 * ZtX) * bread1
        else
            # Homoskedastic SEs
            σ2 = dot(res1, res1) / (n_obs - n_reg)
            vcov = σ2 * bread1
        end
    end

    # J-test computation (Sargan/Hansen)
    residuals = y - X * β
    fitted = X * β
    Zte = Z' * residuals
    j_stat = (Zte' * W * Zte)[1]

    # Degrees of freedom for J-test = Instruments - Regressors
    df_j = size(Z, 2) - length(β)
    j_pval = df_j > 0 ? 1.0 - cdf(Chisq(df_j), j_stat) : NaN

    # Compile metadata
    metadata = Dict{Symbol,Any}(
        :method => "$(steps)-step $(typeof(model_type))",
        :formula => get(diff_data, :formula, "N/A"),
        :panel_info => diff_data.panel_info,
        :ar_tests => Dict{Int,DynamicPanelTest}(),
    )

    # Return results
    return DynamicPanelResult(
        Vector(β),
        Matrix(vcov),
        Vector(residuals),
        Vector(fitted),
        diff_data.n_obs,
        diff_data.n_groups,
        size(Z, 2),
        coef_names,
        Matrix(X),
        Vector(y),
        Z,
        Matrix(W),
        j_stat,
        j_pval,
        is_windmeijer,
        metadata;
        robust=robust,
    )
end

"""
    calculate_clustered_weight_matrix(Z, residuals, panel_info, n_groups)

Compute the clustered robust moment matrix for panel data:

    A = Z' Ω Z

where Ω accounts for clustering within groups.

# Arguments
- `Z::AbstractMatrix`: Instrument or regressor matrix.
- `residuals::AbstractVector`: Residuals from the model.
- `panel_info::Vector{<:NamedTuple}`: Metadata with `.id` indicating group membership.
- `n_groups::Integer`: Number of clusters/groups.

# Returns
- `A::Matrix`: Clustered robust moment matrix of size `(size(Z,2), size(Z,2))`.
"""
function calculate_clustered_weight_matrix(Z, residuals, panel_info, n_groups)
    # Initialize moment matrix
    n_inst = size(Z, 2)
    A = zeros(n_inst, n_inst)
    n_rows = length(panel_info)

    # Return zero matrix if no data
    if n_rows == 0
        return A
    end

    # Tracking indices for group blocks
    start_idx = 1
    current_id = panel_info[1].id

    # Iterate over data to find group blocks
    for i in 1:n_rows
        # Check if end of group
        is_last = (i == n_rows)
        next_id = is_last ? nothing : panel_info[i + 1].id

        # When group ends, compute contribution
        if is_last || (current_id != next_id)
            # Compute for group
            end_idx = i
            Z_i = Z[start_idx:end_idx, :]
            u_i = residuals[start_idx:end_idx]
            Zu_i = Z_i' * u_i
            A += Zu_i * Zu_i'

            # Update for next group
            if !is_last
                current_id = next_id
                start_idx = i + 1
            end
        end
    end

    return A
end

"""
    calculate_windmeijer_correction(Z, X, res1, β2, ZtX, Zty, W2, V2, V1r, panel_info)

Compute the Windmeijer (2005) finite-sample corrected variance-covariance matrix
for a two-step (efficient) linear GMM estimator, following the construction used
in Stata's `xtabond2`/`ivreg2`.

The naive two-step variance `V2 = bread2` ignores that the second-step weighting
matrix `W2 = inv(Ω1)` is itself estimated from first-step residuals `res1`, which
are functions of the estimated first-step coefficients `β1`. Windmeijer's
correction propagates this extra source of estimation uncertainty.

Let `û2 = y - X β2` be the two-step residuals and `c = W2 (Z'y - Z'X β2) = W2 Z'û2`.
The `k`-th column of the derivative matrix `D = ∂β2/∂β1` is

    D[:, k] = -V2 (Z'X)' W2 (∂Ω1/∂β1_k) c

where, crucially, `Ω1 = Σ_i (Z_i'u_i)(Z_i'u_i)'` is a sum over *clusters*, so its
derivative preserves the per-cluster structure:

    ∂Ω1/∂β1_k = -Σ_i [ (Z_i'X_i[:, k])(Z_i'u_i)' + (Z_i'u_i)(Z_i'X_i[:, k])' ].

The corrected variance is then

    V_corr = V2 + D V2 + V2 D' + D V1r D'

with `V1r` the one-step *robust* covariance. (Using the outer product of the
summed moments, or `Z'X β2` in place of `Z'û2`, badly inflates the correction —
both are common pitfalls.)

# Arguments
- `Z::AbstractMatrix`: Instrument matrix (rows grouped by cluster, matching `panel_info`).
- `X::AbstractMatrix`: Regressor matrix.
- `res1::AbstractVector`: First-step residuals.
- `β2::AbstractVector`: Second-step coefficient estimates.
- `ZtX`, `Zty`: Precomputed `Z'X` and `Z'y`.
- `W2::AbstractMatrix`: Second-step weighting matrix `inv(Ω1)`.
- `V2::AbstractMatrix`: Naive two-step covariance (`bread2`).
- `V1r::AbstractMatrix`: One-step robust covariance.
- `panel_info`: Row metadata with `.id`, used to delimit clusters.

# Returns
- `V_corr::Matrix`: Windmeijer-corrected variance-covariance matrix.
"""
function calculate_windmeijer_correction(Z, X, res1, β2, ZtX, Zty, W2, V2, V1r, panel_info)
    n_reg = size(X, 2)
    n_inst = size(Z, 2)
    n_rows = length(panel_info)

    # c = W2 Z'û2, using the two-step residuals (small by the moment conditions)
    c = W2 * (Zty - ZtX * β2)

    # Densify once: row-slicing a SparseMatrixCSC per cluster is far more
    # expensive (each slice re-copies into a new sparse structure) than a single
    # upfront conversion followed by cheap `@view`s into a dense matrix.
    Zd = Z isa AbstractSparseMatrix ? Matrix(Z) : Z

    # Accumulate temp = (∂Ω1/∂β1) applied to c, preserving per-cluster structure.
    # Column k gathers -Σ_i [ (Z_i'u_i · c) Z_i'X_i[:,k] + (Z_i'X_i[:,k] · c) Z_i'u_i ].
    temp = zeros(n_inst, n_reg)
    start_idx = 1
    current_id = panel_info[1].id
    for i in 1:n_rows
        is_last = (i == n_rows)
        next_id = is_last ? nothing : panel_info[i + 1].id
        if is_last || (current_id != next_id)
            Z_i = @view Zd[start_idx:i, :]
            X_i = @view X[start_idx:i, :]
            u_i = @view res1[start_idx:i]
            Zu = Z_i' * u_i           # L
            ZX = Z_i' * X_i           # L×K
            temp .+= -((Zu' * c) .* ZX .+ Zu * (ZX' * c)')
            if !is_last
                current_id = next_id
                start_idx = i + 1
            end
        end
    end

    # D = ∂β2/∂β1 (K×K); assemble the corrected variance.
    D = -V2 * ZtX' * W2 * temp
    V_corr = V2 + D * V2 + V2 * D' + D * V1r * D'
    return Matrix((V_corr + V_corr') / 2)
end

"""
Ensure a matrix is positive definite by adjusting its eigenvalues below a tolerance.

# Arguments
- `A::AbstractMatrix`: The input square matrix.
- `tol::Real=1e-12`: Minimum allowed eigenvalue; eigenvalues below this are shifted.

# Returns
- `Matrix`: A matrix guaranteed to be positive definite.
"""
function posdef_fix(A; tol=1e-12)
    # Compute eigenvalues
    A_sym = Symmetric(A)
    evals = eigvals(A_sym)
    min_ev = minimum(real.(evals))

    # Adjust if necessary
    if min_ev < tol
        return A + I * (tol - min_ev + 1e-10)
    end

    return A
end

"""
Compute the initial (one-step) GMM weighting matrix `(Z'Z)^-1`.

This is the standard choice for all supported estimators (Difference, System,
Anderson-Hsiao); it is a reasonable default that does not depend on unknown
parameters. `diff_data` is accepted for API symmetry but currently unused.

# Arguments
- `model::AbstractDynamicPanelModel`: The dynamic panel model.
- `Z`: Instrument matrix.
- `diff_data`: Differenced data (currently unused).

# Returns
- `W::Matrix`: Initial weighting matrix `(Z'Z)^-1`.
"""
function initial_weight_matrix(::AbstractDynamicPanelModel, Z, diff_data)
    return inv(posdef_fix(Matrix(Z' * Z)))
end
