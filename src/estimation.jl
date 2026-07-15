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

    # Collinear regressors can't be silently dropped like a redundant
    # instrument column, so this errors instead of calling `_drop_collinear_columns`.
    _check_full_rank(X, coef_names)

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
    _qr_rank(F; tol=1e-9) -> Int

Numerical rank from a pivoted QR factorization `F` (as produced by
`qr(M, ColumnNorm())`): the number of `|diag(F.R)|` entries exceeding `tol`
relative to the largest. Shared by `_drop_collinear_columns` (instrument
matrix `Z`: drop the redundant columns) and `estimate`'s regressor check
(matrix `X`: collinearity can't be silently dropped, so it errors instead) —
both apply the same rank-revealing-QR/tolerance convention.
"""
function _qr_rank(F; tol::Real=1e-9)
    d = abs.(diag(F.R))
    isempty(d) && return 0
    return count(>(tol * maximum(d)), d)
end

"""
    _check_full_rank(X, coef_names; tol=1e-9)

Ensure regressor matrix `X` has full column rank, via the same rank-revealing-QR
tolerance convention as `_drop_collinear_columns` (used below for the instrument
matrix `Z`). Collinear regressors can't be silently dropped like a redundant
instrument column — a regressor's coefficient is a target of estimation, not a
disposable moment condition — so this errors instead, naming the offending
columns, rather than surfacing an opaque `LinearAlgebra.SingularException`
deep inside the GMM solver.
"""
function _check_full_rank(X, coef_names; tol::Real=1e-9)
    n_regressors = size(X, 2)
    x_rank = _qr_rank(qr(X, ColumnNorm()); tol=tol)
    if x_rank < n_regressors
        error(
            "Regressors are collinear (rank $x_rank < $n_regressors regressors: " *
            "$(coef_names)). Remove or combine the collinear regressor(s).",
        )
    end
end

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
    r = _qr_rank(F; tol=tol)
    r == 0 && return Z                     # degenerate: nothing to check rank against
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
    G1 = ZtX' * W1  # reused for bread1, β1, and (if one-step robust) the sandwich below
    bread1 = inv(posdef_fix(G1 * ZtX))
    β1 = bread1 * (G1 * Zty)
    res1 = y - X * β1
    β, W, final_bread = β1, W1, bread1
    is_windmeijer = false

    # Computed once and reused by both calculate_clustered_weight_matrix and
    # calculate_windmeijer_correction below (both would otherwise re-walk the
    # same panel_info to find the same cluster boundaries).
    ranges = _cluster_ranges(diff_data.panel_info)

    # Densified at most once and threaded through every call below:
    # calculate_clustered_weight_matrix and calculate_windmeijer_correction
    # would otherwise each independently call Matrix(Z) on the same sparse
    # instrument matrix, paying the conversion twice instead of once. Only
    # computed when actually needed (skipped for one-step, non-robust fits).
    Zd = (steps == 2 || robust) ? _densify(Z) : Z

    # Two-Step GMM
    if steps == 2
        # Second-Step Estimation
        Ω_clustered = calculate_clustered_weight_matrix(
            Zd, res1, diff_data.panel_info, diff_data.n_groups; ranges=ranges
        )
        W2 = inv(posdef_fix(Ω_clustered))
        bread2 = inv(posdef_fix(ZtX' * W2 * ZtX))
        β2 = bread2 * (ZtX' * W2 * Zty)
        β, W, final_bread = β2, W2, bread2

        # Variance-Covariance Matrix
        if robust && windmeijer
            # One-step robust (sandwich) covariance, used in the Windmeijer correction
            V1r = bread1 * (G1 * Ω_clustered * G1') * bread1
            # Windmeijer Finite-Sample Correction
            vcov = calculate_windmeijer_correction(
                Zd, X, res1, β2, ZtX, Zty, W2, bread2, V1r, diff_data.panel_info; ranges=ranges
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
                Zd, res1, diff_data.panel_info, diff_data.n_groups; ranges=ranges
            )
            vcov = bread1 * (G1 * Ω_clustered * G1') * bread1
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
# Densify once: row-slicing a SparseMatrixCSC per cluster/individual is far more
# expensive (each slice re-copies into a new sparse structure) than a single
# upfront conversion followed by cheap `@view`s into a dense matrix.
_densify(Z) = Z isa AbstractSparseMatrix ? Matrix(Z) : Z

# Ranges of consecutive rows sharing the same `.id`, assuming `panel_info` is
# already grouped by individual (an existing invariant of the pipeline).
function _cluster_ranges(panel_info)
    n_rows = length(panel_info)
    n_rows == 0 && return UnitRange{Int}[]
    ranges = UnitRange{Int}[]
    start_idx = 1
    current_id = panel_info[1].id
    for i in 1:n_rows
        is_last = (i == n_rows)
        next_id = is_last ? nothing : panel_info[i + 1].id
        if is_last || current_id != next_id
            push!(ranges, start_idx:i)
            if !is_last
                current_id = next_id
                start_idx = i + 1
            end
        end
    end
    return ranges
end

function calculate_clustered_weight_matrix(
    Z, residuals, panel_info, n_groups; ranges=_cluster_ranges(panel_info)
)
    n_inst = size(Z, 2)
    A = zeros(n_inst, n_inst)
    Zd = _densify(Z)
    for r in ranges
        Zu_i = @view(Zd[r, :])' * @view(residuals[r])
        mul!(A, Zu_i, Zu_i', true, true)   # A += Zu_i * Zu_i', fused, no temporary
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
function calculate_windmeijer_correction(
    Z, X, res1, β2, ZtX, Zty, W2, V2, V1r, panel_info; ranges=_cluster_ranges(panel_info)
)
    n_reg = size(X, 2)
    n_inst = size(Z, 2)

    # c = W2 Z'û2, using the two-step residuals (small by the moment conditions)
    c = W2 * (Zty - ZtX * β2)

    Zd = _densify(Z)

    # Accumulate temp = (∂Ω1/∂β1) applied to c, preserving per-cluster structure.
    # Column k gathers -Σ_i [ (Z_i'u_i · c) Z_i'X_i[:,k] + (Z_i'X_i[:,k] · c) Z_i'u_i ].
    temp = zeros(n_inst, n_reg)
    for r in ranges
        Z_i = @view Zd[r, :]
        X_i = @view X[r, :]
        u_i = @view res1[r]
        Zu = Z_i' * u_i           # L
        ZX = Z_i' * X_i           # L×K
        s = dot(Zu, c)
        w = ZX' * c
        temp .-= s .* ZX
        mul!(temp, Zu, w', -1.0, 1.0)   # temp -= Zu * w', fused, no temporary
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
Compute the initial (one-step) GMM weighting matrix.

For `DifferenceGMM`, this is `(Z'HZ)^-1` (Arellano & Bond, 1991, p. 279), where
`H` reflects the MA(1) structure that first-differencing imposes on a
homoskedastic error term: `H_i[t,s] = 2` if `t == s`, `-1` if periods `t` and
`s` are calendar-adjacent, `0` otherwise, summed over each individual's own
differenced-equation rows (handling gaps in unbalanced panels, where
non-adjacent rows get no off-diagonal term). Using plain `(Z'Z)^-1` instead is
a common simplification but is not the efficient one-step weighting matrix
under homoskedasticity and biases the Sargan/Hansen J-statistic downward.

For `SystemGMM` and `AndersonHsiao`, the plain `(Z'Z)^-1` is used: the `H`
matrix above is specific to the pure-difference moment structure and does not
carry over unchanged once level-equation moments are added.

# Arguments
- `model::AbstractDynamicPanelModel`: The dynamic panel model.
- `Z`: Instrument matrix.
- `diff_data`: Differenced data; `panel_info` is used for `DifferenceGMM`.

# Returns
- `W::Matrix`: Initial weighting matrix.
"""
function initial_weight_matrix(::AbstractDynamicPanelModel, Z, diff_data)
    return inv(posdef_fix(Matrix(Z' * Z)))
end

function initial_weight_matrix(::DifferenceGMM, Z, diff_data)
    # A_N = N^-1 * sum_i Z_i'HZ_i (Arellano & Bond, 1991, eq. 3-4); the N^-1 factor
    # cancels out of the coefficient estimate but is required for the one-step
    # Sargan/Hansen J-statistic to have its asymptotic chi-squared scale.
    N = diff_data.n_groups
    return inv(posdef_fix(_ab_h_weight_matrix(Z, diff_data.panel_info) / N))
end

# Z'HZ summed over individuals, where H_i[t,s] = 2 (t==s), -1 (calendar-adjacent
# periods), 0 otherwise — restricted to each individual's own differenced-equation
# rows (is_level == false), so unbalanced-panel gaps correctly get no off-diagonal term.
function _ab_h_weight_matrix(Z, panel_info)
    n_inst = size(Z, 2)
    A = zeros(n_inst, n_inst)
    Zd = _densify(Z)
    ranges = _cluster_ranges(panel_info)

    # Z_i'*H_i (n_inst × m) is recomputed fresh every cluster; preallocate one
    # buffer sized to the largest cluster and reuse it via a view, rather than
    # letting `mul!` below allocate a new temporary on every iteration.
    maxm = maximum(r -> count(j -> !get(panel_info[j], :is_level, false), r), ranges; init=0)
    buf = zeros(n_inst, maxm)

    for r in ranges
        rows = [(j, panel_info[j].time) for j in r if !get(panel_info[j], :is_level, false)]
        isempty(rows) && continue
        idxs = first.(rows)
        times = last.(rows)
        # `idxs` is a filtered Vector{Int}, not a UnitRange, so a view here
        # would be a non-strided SubArray that can't dispatch to BLAS gemm for
        # `Z_i' * H_i * Z_i` below — materializing is faster despite the copy
        # (benchmarked: ~2x faster than @view at realistic cluster/instrument sizes).
        Z_i = Zd[idxs, :]
        m = length(idxs)
        H_i = zeros(m, m)
        for a in 1:m
            H_i[a, a] = 2.0
            for b in (a + 1):m
                if times[b] - times[a] == 1
                    H_i[a, b] = H_i[b, a] = -1.0
                end
            end
        end
        buf_view = @view buf[:, 1:m]
        mul!(buf_view, Z_i', H_i)              # buf := Z_i'*H_i, reused across iterations
        mul!(A, buf_view, Z_i, true, true)      # A += buf*Z_i, fused final product
    end
    return A
end
