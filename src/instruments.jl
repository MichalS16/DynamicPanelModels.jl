# src/instruments.jl

"""
    build_instruments(model::AbstractDynamicPanelModel, diff_data::NamedTuple; kwargs...)

Construct the instrument matrix for a dynamic panel model.

# Arguments
- `model::AbstractDynamicPanelModel`: Estimator object (e.g., `DifferenceGMM`, `SystemGMM`).
- `diff_data::NamedTuple`: Differenced or transformed panel data.
- `kwargs`: Optional keyword arguments:
    - `collapse::Bool=false`: Collapse instruments to one column per lag.
    - `max_lags::Int=Inf`: Maximum number of lags to include as instruments.

# Returns
- `AbstractMatrix`: Instrument matrix specific to the model.
"""
function build_instruments(model::AbstractDynamicPanelModel, diff_data; kwargs...)
    return error("Instrument generation not implemented for $(typeof(model)).")
end

"""
    append_exog_instruments(Z, diff_data; is_level::Bool=false)

Append "IV-style" instrument columns for strictly exogenous regressors
(`diff_data.exog_idx`, set via the `exog` keyword of `get_diff_data`/`fit`).
Each exogenous regressor already appears directly in `X`; per standard GMM
dynamic panel practice (Roodman, 2009), it is additionally used as its own
instrument (in the same transformed form — differenced or level — as it
enters `X`) to exploit its exogeneity and improve identification/efficiency
of its coefficient.

Only rows matching `is_level` (i.e. `row.is_level == is_level`) get a nonzero
value; other rows are zero, preserving the block-diagonal structure expected
between the difference-equation and level-equation instrument blocks.

Returns `Z` unchanged if `diff_data` has no `exog_idx` field or it is empty.
"""
function append_exog_instruments(Z, diff_data; is_level::Bool=false)
    exog_idx = get(diff_data, :exog_idx, Int[])
    isempty(exog_idx) && return Z

    n_obs = size(Z, 1)
    row_mask = [get(row, :is_level, false) == is_level for row in diff_data.panel_info]
    exog_block = zeros(n_obs, length(exog_idx))
    exog_block[row_mask, :] = diff_data.X[row_mask, exog_idx]

    return hcat(Z, exog_block)
end

# Map each panel time period to its 1-based index in the sorted `valid_times`.
_make_time_map(all_times) = Dict(t => i for (i, t) in enumerate(all_times))

# Look up `lag_time` for individual `c_id` in `values_by_id` (a Dict{id, Dict{time, val}}
# as built by get_diff_data); if present and nonzero/non-NaN, push (row, col, value) into
# the sparse-matrix triplet vectors. Shared by the three build_instruments methods below,
# which otherwise repeat this exact guard.
function _push_instrument!(rows, cols, vals, values_by_id, c_id, lag_time, i, col)
    haskey(values_by_id, c_id) || return nothing
    indiv_data = values_by_id[c_id]
    haskey(indiv_data, lag_time) || return nothing
    val = indiv_data[lag_time]
    if !isnan(val) && val != 0.0
        push!(rows, i)
        push!(cols, col)
        push!(vals, val)
    end
    return nothing
end

"""
    build_instruments(model::DifferenceGMM, diff_data::NamedTuple; collapse=false, max_lags=999)

Construct the Arellano-Bond (1991) instrument matrix for Difference GMM.

# Arguments
- `model::DifferenceGMM`: Difference GMM estimator.
- `diff_data::NamedTuple`: Differenced panel data containing:
    - `panel_info`: Vector of observation metadata (`id`, `time`, optional `is_level`).
    - `n_obs`: Total number of observations.
    - `id_time_to_y`: Dict mapping `(id, time)` to response values.
    - `valid_times`: Vector of time periods.
- `collapse::Bool=false`: Collapse instruments to one column per lag.
- `max_lags::Int=999`: Maximum number of lags (count) per period.
- `min_lag::Int=1`: Minimum instrument lag order.
- `max_lag::Int=typemax(Int)`: Maximum instrument lag order.

# Returns
- `SparseMatrixCSC{Float64}`: Sparse instrument matrix for GMM estimation.
"""
function build_instruments(
    model::DifferenceGMM,
    diff_data::NamedTuple;
    collapse::Bool=false,
    max_lags::Int=999,
    min_lag::Int=1,
    max_lag::Int=typemax(Int),
)
    Z = _build_ab_instruments(
        diff_data; collapse=collapse, max_lags=max_lags, min_lag=min_lag, max_lag=max_lag
    )
    return append_exog_instruments(Z, diff_data; is_level=false)
end

"""
    _build_ab_instruments(diff_data; collapse=false, max_lags=999, min_lag=1, max_lag=typemax(Int))

Build the standard Arellano-Bond GMM-style instrument block (lagged levels of
`y` as instruments for the transformed equation), without appending any
"IV-style" exogenous-regressor instrument columns. Shared by the `DifferenceGMM`
and `SystemGMM` methods of `build_instruments`.

`min_lag`/`max_lag` bound the instrument lag *order* (relative to the current
period), and `max_lags` caps the *count* of lags per period (shallowest first) —
together giving `xtabond2`-style lag limits.
"""
function _build_ab_instruments(
    diff_data::NamedTuple;
    collapse::Bool=false,
    max_lags::Int=999,
    min_lag::Int=1,
    max_lag::Int=typemax(Int),
)
    min_lag > max_lag && error("min_lag ($min_lag) must be <= max_lag ($max_lag).")
    max_lags < 1 && error("max_lags must be >= 1, got $max_lags.")

    # Unpack diff_data
    panel_info = diff_data.panel_info
    n_obs = diff_data.n_obs
    id_time_to_y = diff_data.id_time_to_y
    all_times = diff_data.valid_times
    time_map = _make_time_map(all_times)
    T_max = length(all_times)

    # Instrument timing depends on the transform: first differences use lags from
    # y_{t-2} (offset 1), forward orthogonal deviations from y_{t-1} (offset 0).
    # The lag *order* of the `l`-th instrument is `offset + l`.
    offset = get(diff_data, :transform, :fd) == :fod ? 0 : 1
    first_t = 2 + offset  # earliest period with a usable lag instrument

    # If too short panel, return empty instrument matrix
    if T_max < first_t
        return spzeros(n_obs, 0)
    end

    # `l` positions used for a period at index `t`, honoring min_lag/max_lag/max_lags.
    # The valid set is always a contiguous range (order = offset + l is monotone in
    # l, and min_lag/max_lag/max_lags only truncate it), so this returns a UnitRange
    # with no allocation — important since it's called once per observation.
    function used_ls(t)
        avail = t - 1 - offset
        lo = max(1, min_lag - offset)
        hi = min(avail, max_lag - offset, lo + max_lags - 1)
        return lo:hi
    end

    # Column layout
    base = max(offset + 1, min_lag)  # shallowest included lag order (collapse indexing)
    col_offsets = Dict{Int,Int}()
    if collapse
        max_order = min(T_max - 1, max_lag)
        n_cols = max_order >= base ? min(max_lags, max_order - base + 1) : 0
    else
        curr = 1
        for t in first_t:T_max
            col_offsets[t] = curr
            curr += length(used_ls(t))
        end
        n_cols = curr - 1
    end

    # Build sparse matrix entries
    rows, cols, vals = Int[], Int[], Float64[]
    for (i, row_info) in enumerate(panel_info)
        get(row_info, :is_level, false) && continue
        c_id, c_time = row_info.id, row_info.time
        t_idx = time_map[c_time]
        t_idx < first_t && continue

        for (pos, l) in enumerate(used_ls(t_idx))
            lag_time_val = all_times[t_idx - offset - l]
            target_col = collapse ? ((offset + l) - base + 1) : (col_offsets[t_idx] + pos - 1)
            _push_instrument!(rows, cols, vals, id_time_to_y, c_id, lag_time_val, i, target_col)
        end
    end

    return sparse(rows, cols, vals, n_obs, max(n_cols, 0))
end

"""
    build_instruments(model::SystemGMM, diff_data::NamedTuple; collapse::Bool=false, max_lags::Int=Inf)

Construct the System GMM (Blundell-Bond) instrument matrix by combining:

1. Difference equation instruments (lags of Δy)
2. Level equation instruments (Δy_{t-1} for t ≥ 3)

# Arguments
- `model::SystemGMM`: System GMM estimator object.
- `diff_data::NamedTuple`: Panel data in differenced form with:
    - `panel_info`: Vector of observation metadata (`id`, `time`, `is_level`).
    - `n_obs`: Total number of observations.
    - `id_time_to_diff_y`: Dict mapping `id => Dict(time => Δy)` for level instruments.
    - `valid_times`: Vector of time periods.
- `collapse::Bool=false`: Whether to collapse instruments across periods.
- `max_lags::Int=Inf`: Maximum number of lags for difference instruments.

# Returns
- `SparseMatrixCSC{Float64}`: Combined instrument matrix for difference and level equations.
"""
function build_instruments(
    model::SystemGMM,
    diff_data::NamedTuple;
    collapse::Bool=false,
    max_lags::Int=999,
    min_lag::Int=1,
    max_lag::Int=typemax(Int),
)
    # Build Difference Equation Instruments
    Z_diff = _build_ab_instruments(
        diff_data; collapse=collapse, max_lags=max_lags, min_lag=min_lag, max_lag=max_lag
    )
    Z_diff = append_exog_instruments(Z_diff, diff_data; is_level=false)
    panel_info = diff_data.panel_info
    n_total_obs = diff_data.n_obs
    id_time_to_diff_y = diff_data.id_time_to_diff_y
    all_times = diff_data.valid_times
    time_map = _make_time_map(all_times)
    T_max = length(all_times)

    # If too short panel, return difference instruments only
    if T_max < 3
        return Z_diff
    end

    # Build Level Equation Instruments
    n_level_cols = collapse ? 1 : (T_max - 2)
    rows_lev, cols_lev, vals_lev = Int[], Int[], Float64[]

    # Loop over observations
    for (i, row_info) in enumerate(panel_info)
        # Only level observations
        !get(row_info, :is_level, false) && continue
        c_id, c_time = row_info.id, row_info.time
        t_idx = time_map[c_time]

        # Only t >= 3 have level instruments
        t_idx < 3 && continue
        prev_time_val = all_times[t_idx - 1]
        target_col = collapse ? 1 : (t_idx - 2)
        _push_instrument!(
            rows_lev, cols_lev, vals_lev, id_time_to_diff_y, c_id, prev_time_val, i, target_col
        )
    end

    # Construct sparse matrix for level instruments
    Z_level = sparse(rows_lev, cols_lev, vals_lev, n_total_obs, n_level_cols)
    Z_level = append_exog_instruments(Z_level, diff_data; is_level=true)
    return hcat(Z_diff, Z_level)
end

"""
    build_instruments(model::AndersonHsiao, diff_data::NamedTuple; kwargs...)

Construct the instrument matrix for the Anderson-Hsiao estimator.

# Arguments
- `model::AndersonHsiao`: Anderson-Hsiao dynamic panel model object.
- `diff_data::NamedTuple`: Preprocessed panel data from `get_diff_data`. Any regressors
  named in `exog` (see `get_diff_data`) are appended as additional instrument columns.
- `kwargs...`: Additional keyword arguments (currently ignored).

# Returns
- `SparseMatrixCSC{Float64, Int}`: Instrument matrix with one column of `y_{i,t-2}` for
  each differenced observation, plus one column per strictly exogenous regressor.
"""
function build_instruments(model::AndersonHsiao, diff_data::NamedTuple; kwargs...)
    # Unpack diff_data
    panel_info = diff_data.panel_info
    n_obs = diff_data.n_obs
    id_time_to_y = diff_data.id_time_to_y
    all_times = diff_data.valid_times
    time_map = _make_time_map(all_times)
    rows, cols, vals = Int[], Int[], Float64[]

    # Loop over observations
    for (i, row_info) in enumerate(panel_info)
        # Only differenced observations
        get(row_info, :is_level, false) && continue
        c_id, c_time = row_info.id, row_info.time
        t_idx = time_map[c_time]
        t_idx < 3 && continue
        lag_2_time = all_times[t_idx - 2]
        _push_instrument!(rows, cols, vals, id_time_to_y, c_id, lag_2_time, i, 1)
    end

    Z = sparse(rows, cols, vals, n_obs, 1)
    return append_exog_instruments(Z, diff_data; is_level=false)
end
