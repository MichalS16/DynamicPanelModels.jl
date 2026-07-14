# src/transformations.jl

"""
    parse_formula(formula::AbstractString) -> Tuple{String, Vector{String}}

Parse a simple regression formula of the form `"y ~ x1 + x2 + ..."`.
Only additive terms are supported; the sole transformation recognized is
`lag(var)` for a lagged regressor. Interactions are not supported. A
`FormulaTerm` from StatsModels' `@formula` is also accepted (see the
`parse_formula(::FormulaTerm)` method).

# Arguments
- `formula::AbstractString`: Regression formula with `~` separating dependent and independent variables, and `+` separating regressors.

# Returns
- `Tuple{String, Vector{String}}`: 
  - First element: dependent variable name
  - Second element: vector of independent variable names in order

# Errors
Throws an error if:
- the formula is empty or malformed,
- `~` is missing or appears more than once,
- no valid regressors are provided,
- duplicate regressors are present,
- the dependent variable appears on the right-hand side.
"""
function parse_formula(formula::AbstractString)
    # Trim whitespace
    clean_formula = strip(formula)
    if isempty(clean_formula)
        error("Formula string cannot be empty. Please provide a valid formula. e.g., 'y ~ x1 + x2'")
    end

    # Check for '~'
    if !occursin("~", clean_formula)
        error("Invalid formula. Missing '~'.")
    end

    # Split into parts and validate parts
    parts = split(clean_formula, "~")
    if length(parts) > 2
        error("Invalid formula. Too many '~' symbols found.")
    end

    # Dependent variable
    y_name = strip(String(parts[1]))
    if isempty(y_name)
        error("Missing dependent variable (left side of ~).")
    end

    # Independent variables
    ind_string = strip(String(parts[2]))
    if isempty(ind_string)
        error("Missing independent variables (right side of ~).")
    end

    # Parse independent variables
    x_names = String[]
    for term in split(ind_string, "+")
        term = strip(String(term))
        if isempty(term)
            continue
        end
        if term in x_names
            error("Duplicate independent variable found: '$term'.")
        end
        push!(x_names, term)
    end

    # Validate independent variables
    if isempty(x_names)
        error("No valid independent variables found.")
    end

    # Check for dependent variable in independent variables
    if y_name in x_names
        error("Variable '$y_name' appears on both sides of the formula.")
    end

    return (y_name, x_names)
end

"""
    lag(x)

Marker for a lagged regressor inside a `@formula`, e.g. `@formula(y ~ lag(y) + x)`.

Lags are resolved per cross-sectional unit during data preparation
(`get_diff_data`), so `lag` is only meaningful symbolically inside a formula and
is never evaluated on data directly. Calling it outside a formula raises an error.
"""
function lag(x)
    return throw(
        ArgumentError(
            "`lag` is a formula marker; use it inside @formula, e.g. @formula(y ~ lag(y) + x)."
        ),
    )
end

# String representation of a single StatsModels term, matching the string-formula syntax.
_term_string(t::Term) = string(t.sym)
_term_string(t::FunctionTerm) = string(t.exorig)

"""
    parse_formula(f::FormulaTerm) -> Tuple{String, Vector{String}}

Parse a StatsModels `@formula` (e.g. `@formula(y ~ lag(y) + x)`) into the same
`(dependent, regressors)` form as the string parser. Intercept terms are dropped
(dynamic panel GMM has no intercept in the differenced equation). Delegates to
the string [`parse_formula`](@ref) for validation, so the same rules apply.
"""
function parse_formula(f::FormulaTerm)
    y_name = _term_string(f.lhs)
    rhs = f.rhs isa Tuple ? f.rhs : (f.rhs,)
    x_terms = String[]
    for t in rhs
        (t isa ConstantTerm || t isa InterceptTerm) && continue
        push!(x_terms, _term_string(t))
    end
    isempty(x_terms) && error("Missing independent variables (right side of ~).")
    return parse_formula("$y_name ~ " * join(x_terms, " + "))
end

# Whitelisted unary transformations usable in formulas (no `eval`, for safety).
const _TRANSFORMS = Dict{String,Function}(
    "log" => log, "log10" => log10, "log2" => log2, "exp" => exp, "sqrt" => sqrt, "abs" => abs
)

# Parse a base expression `var` or `f(var)` into (var::Symbol, label::String, fn::Function).
function _parse_expr(expr::AbstractString)
    e = strip(String(expr))
    m = match(r"^([A-Za-z_][A-Za-z0-9_]*)\(([A-Za-z_][A-Za-z0-9_]*)\)$", e)
    if m !== nothing
        fname, var = m.captures[1], m.captures[2]
        haskey(_TRANSFORMS, fname) || error(
            "Unsupported transformation '$fname'. Supported: " *
            join(sort(collect(keys(_TRANSFORMS))), ", ") *
            ".",
        )
        return (var=Symbol(var), label="$fname($var)", fn=_TRANSFORMS[fname])
    end
    occursin(r"^[A-Za-z_][A-Za-z0-9_]*$", e) || error("Cannot parse formula term '$e'.")
    return (var=Symbol(e), label=e, fn=identity)
end

# Parse a lag spec ("2" or "0:1") into a vector of non-negative lag orders.
function _parse_lagspec(spec::AbstractString)
    s = strip(String(spec))
    lags = if occursin(':', s)
        p = split(s, ':')
        length(p) == 2 || error("Invalid lag range '$s'.")
        collect(parse(Int, strip(p[1])):parse(Int, strip(p[2])))
    else
        [parse(Int, s)]
    end
    any(<(0), lags) && error("Lag orders must be non-negative (got '$s').")
    isempty(lags) && error("Empty lag range '$s'.")
    return lags
end

"""
    _parse_term(term) -> Vector{NamedTuple}

Parse one right-hand-side formula term into one or more specs
`(var::Symbol, label::String, fn::Function, lag::Int)`. Supports:
`var`, `f(var)`, `lag(EXPR)`, `lag(EXPR, k)`, `lag(EXPR, a:b)` where `EXPR` is
`var` or a whitelisted `f(var)`, and `lag(EXPR)` defaults to `lag(EXPR, 1)`.
A range (`a:b`) expands to one spec per lag order.
"""
function _parse_term(term::AbstractString)
    t = strip(String(term))
    m = match(r"^lag\((.*)\)$"s, t)
    if m !== nothing
        inner = strip(m.captures[1])
        ci = findfirst(==(','), inner)  # transforms are unary, so any comma is the lag spec
        expr, lags = if ci === nothing
            (inner, [1])
        else
            (strip(inner[1:(ci - 1)]), _parse_lagspec(inner[(ci + 1):end]))
        end
        base = _parse_expr(expr)
        return [(var=base.var, label=base.label, fn=base.fn, lag=k) for k in lags]
    end
    base = _parse_expr(t)
    return [(var=base.var, label=base.label, fn=base.fn, lag=0)]
end

# Display name for a coefficient given its parsed spec.
function _coef_name(s)
    s.lag == 0 && return s.label
    s.lag == 1 && return "L.$(s.label)"
    return "L$(s.lag).$(s.label)"
end

"""
    get_diff_data(data, id_col::Symbol, time_col::Symbol, formula, model::AbstractDynamicPanelModel; exog=String[])

Prepare panel data for dynamic panel GMM estimation by constructing lagged regressors, applying first differences, and optionally stacking level equations for System GMM. Observations invalid due to lagging or differencing are removed.

# Arguments
- `data`: Panel dataset as any Tables.jl-compatible source (e.g. a `DataFrame`).
- `id_col::Symbol`: Column identifying individuals.
- `time_col::Symbol`: Column identifying time periods.
- `formula`: Model formula as a string (`"y ~ lag(y) + x"`) or a StatsModels
  `@formula`. Supported term syntax:
    - `lag(var)` / `lag(var, k)` — first / `k`-th lag; `lag(var, a:b)` expands to one
      regressor per lag order in the range (e.g. `lag(x, 0:1)` → `x` and `L.x`).
    - Whitelisted transforms `log, log10, log2, exp, sqrt, abs`, usable on either side
      and nested in a lag, e.g. `log(y) ~ lag(log(y)) + log(x)`.
- `model::AbstractDynamicPanelModel`: GMM estimator type (Difference or System GMM).
- `exog`: Names of right-hand-side regressors (as they appear in `coef_names`, e.g. `"x1"`
  or `"L.x1"`) that are strictly exogenous. These are appended to the instrument matrix as
  their own ("IV-style") instrument column, following standard practice for GMM dynamic
  panel estimators (Roodman, 2009); this is in addition to using them directly as
  regressors. The dependent variable (or its lag) may not be marked exogenous.
- `time_effects`: If `true`, add `T-2` period dummies (the first two periods are the
  reference, for identification in the differenced equation) as exogenous regressors.

# Returns
A `NamedTuple` with:
- `y`: Response vector.
- `X`: Regressor matrix.
- `coef_names`: Names of coefficients aligned with `X`.
- `panel_info`: `(id, time, is_level)` metadata per observation.
- `n_obs`: Number of valid observations.
- `n_groups`: Number of individuals/groups.
- `id_time_to_y`: Mapping of level values for instruments.
- `id_time_to_diff_y`: Mapping of differenced values (System GMM).
- `valid_times`: Sorted vector of panel time periods.
- `exog_idx`: Column indices of `X` (aligned with `coef_names`) that are strictly exogenous.
"""
function get_diff_data(
    data,
    id_col::Symbol,
    time_col::Symbol,
    formula::Union{AbstractString,FormulaTerm},
    model::AbstractDynamicPanelModel;
    exog::AbstractVector{<:AbstractString}=String[],
    time_effects::Bool=false,
    transform::Symbol=:fd,
)
    transform in (:fd, :fod) || throw(
        ArgumentError(
            "`transform` must be :fd (first difference) or :fod (forward orthogonal deviations).",
        ),
    )
    (transform == :fod && model isa SystemGMM) && throw(
        ArgumentError("Forward orthogonal deviations (:fod) are not supported for System GMM.")
    )
    # Accept any Tables.jl-compatible source; work with a DataFrame internally.
    Tables.istable(data) ||
        throw(ArgumentError("`data` must be a Tables.jl-compatible table (e.g. a DataFrame)."))
    df = data isa DataFrame ? data : DataFrame(data)

    # Input Validations
    if nrow(df) == 0
        error("Input DataFrame is empty.")
    end
    if !(id_col in propertynames(df))
        error("ID column ':$id_col' not found.")
    end
    if !(time_col in propertynames(df))
        error("Time column ':$time_col' not found.")
    end
    # Duplicate (id, time) rows silently corrupt the lag/difference construction
    # below (arbitrary sub-ordering within a group), so reject them up front.
    # `allunique` on the zip iterator avoids materializing an intermediate vector.
    if !allunique(zip(df[!, id_col], df[!, time_col]))
        error(
            "Duplicate (id, time) pairs found in `data`. Each individual/period " *
            "combination must be unique; check for duplicated rows before fitting.",
        )
    end

    # Parse formula into a response spec and (range-expanded) regressor specs.
    y_name, x_terms = parse_formula(formula)
    y_spec = only(_parse_term(y_name))
    y_spec.lag == 0 || error("The dependent variable cannot be lagged.")
    x_specs = isempty(x_terms) ? NamedTuple[] : reduce(vcat, _parse_term.(x_terms))

    # Check base-variable existence & numeric type
    base_vars = unique([y_spec.var; [s.var for s in x_specs]])
    for v in base_vars
        v in propertynames(df) || error("Variable '$v' missing in DataFrame.")
        eltype(df[!, v]) <: Number || error("Variable '$v' must be numeric.")
    end

    # Coefficient names from the parsed specs
    coef_names = String[_coef_name(s) for s in x_specs]

    # Resolve exogenous regressors: match against coefficient names. Any regressor
    # (including a lag of a strictly exogenous covariate) may be exogenous, but the
    # dependent variable and its lags may not.
    exog_idx = Int[]
    for e in exog
        idx = findfirst(==(e), coef_names)
        idx === nothing &&
            error("Exogenous regressor '$e' is not a right-hand-side term in the formula.")
        x_specs[idx].var == y_spec.var &&
            error("'$e' is (a lag/transform of) the dependent variable and cannot be exogenous.")
        push!(exog_idx, idx)
    end

    # Sort DataFrame by ID and Time
    df_sorted = sort(df, [id_col, time_col])
    ids = unique(df_sorted[!, id_col])
    valid_times = sort(unique(df_sorted[!, time_col]))

    # Time-effect dummies, treated as exogenous regressors instrumented by
    # themselves. In a first-differenced equation only T-2 period dummies are
    # identified (differencing costs one degree of freedom beyond the usual
    # reference period), so the first two periods are dropped.
    dummy_periods = time_effects ? valid_times[3:end] : eltype(valid_times)[]
    n_x = length(x_specs)
    n_dummy = length(dummy_periods)
    n_cols = n_x + n_dummy
    for p in dummy_periods
        push!(coef_names, "t=$(p)")
    end
    append!(exog_idx, (n_x + 1):(n_x + n_dummy))

    # Concrete id/time element types keep the lookups and panel metadata
    # type-stable (avoids boxing identifiers as `Any` in the hot data-prep loop).
    ID = eltype(ids)
    TIME = eltype(valid_times)

    # Initialize Lookups
    id_time_to_y = Dict{ID,Dict{TIME,Float64}}()
    id_time_to_diff_y = Dict{ID,Dict{TIME,Float64}}()

    # Prepare Storage
    y_final = Float64[]
    X_final = Vector{Float64}[]
    panel_info = NamedTuple{(:id, :time, :is_level),Tuple{ID,TIME,Bool}}[]

    # Iterate Groups. `df_sorted` is sorted by (id, time), so `groupby` yields
    # contiguous per-individual SubDataFrames in one pass (no per-group full-table
    # masking), with rows already ordered by time within each group.
    for id_data in groupby(df_sorted, id_col; sort=true)
        times = id_data[!, time_col]
        n_t = length(times)

        # Skip if too short
        if n_t < 3
            continue
        end

        # Group identifier (homogeneous within the group)
        id = id_data[1, id_col]

        # Extract (possibly transformed) response levels
        y_vals = y_spec.fn.(Float64.(id_data[!, y_spec.var]))

        # Build regressor levels: apply the transform, then shift by the lag order.
        X_vals = zeros(n_t, n_cols)
        for (j, s) in enumerate(x_specs)
            col = s.fn.(Float64.(id_data[!, s.var]))
            if s.lag == 0
                X_vals[:, j] = col
            elseif s.lag < n_t
                X_vals[(s.lag + 1):end, j] = col[1:(end - s.lag)]
                X_vals[1:(s.lag), j] .= NaN
            else
                X_vals[:, j] .= NaN   # lag deeper than the group: no valid values
            end
        end

        # Time-effect dummy columns (levels; differenced along with the rest)
        for (jd, p) in enumerate(dummy_periods)
            X_vals[:, n_x + jd] = Float64.(times .== p)
        end

        # Store Level Values Lookup
        group_y_map = Dict{TIME,Float64}()
        for (k, t) in enumerate(times)
            group_y_map[t] = y_vals[k]
        end
        id_time_to_y[id] = group_y_map

        # Transformed equation: first differences (:fd) or forward orthogonal
        # deviations (:fod, Arellano-Bover 1995).
        group_diff_map = Dict{TIME,Float64}()
        if transform == :fd
            for k in 2:n_t
                dy = y_vals[k] - y_vals[k - 1]
                dX = @views X_vals[k, :] - X_vals[k - 1, :]
                if !isnan(dy) && !any(isnan, dX)
                    push!(y_final, dy)
                    push!(X_final, dX)
                    push!(panel_info, (id=id, time=times[k], is_level=false))
                    group_diff_map[times[k]] = dy
                end
            end
        else  # :fod — subtract the mean of all future observations, then scale.
            # Suffix sums turn each mean(fut) into an O(1) lookup instead of an
            # O(T-k) reduction, avoiding an O(T^2) pass per individual. NaN still
            # propagates through the running sum exactly as it would through a
            # fresh `mean` call, so a lagged-regressor NaN in the tail correctly
            # NaNs out the mean (and hence the observation) for earlier k, too.
            y_suffix = cumsum(@view y_vals[end:-1:1])[end:-1:1]  # y_suffix[k] = sum(y_vals[k:end])
            X_suffix = cumsum(@view X_vals[end:-1:1, :]; dims=1)[end:-1:1, :]
            for k in 1:(n_t - 1)
                Tf = n_t - k
                c = sqrt(Tf / (Tf + 1))
                y_fut_mean = (y_suffix[k + 1]) / Tf
                x_fut_mean = @views (X_suffix[k + 1, :]) ./ Tf
                y_star = c * (y_vals[k] - y_fut_mean)
                x_star = c .* (@view(X_vals[k, :]) .- x_fut_mean)
                if !isnan(y_star) && !any(isnan, x_star)
                    push!(y_final, y_star)
                    push!(X_final, x_star)
                    push!(panel_info, (id=id, time=times[k], is_level=false))
                    group_diff_map[times[k]] = y_star
                end
            end
        end
        id_time_to_diff_y[id] = group_diff_map

        # Level Equation (System GMM Only)
        if model isa SystemGMM
            # Append level observations
            for k in 3:n_t
                lev_y = y_vals[k]
                lev_X = X_vals[k, :]

                # Check validity
                if !isnan(lev_y) && !any(isnan, lev_X)
                    push!(y_final, lev_y)
                    push!(X_final, lev_X)
                    push!(panel_info, (id=id, time=times[k], is_level=true))
                end
            end
        end
    end

    # Finalize
    n_obs = length(y_final)
    if n_obs == 0
        error("No valid observations generated. Check data quality or panel length.")
    end

    # Convert X list to Matrix
    X_mat = zeros(n_obs, n_cols)
    for i in 1:n_obs
        X_mat[i, :] = X_final[i]
    end

    # Return results
    return (
        y=y_final,
        X=X_mat,
        coef_names=coef_names,
        panel_info=panel_info,
        n_obs=n_obs,
        n_groups=length(ids),
        id_time_to_y=id_time_to_y,
        id_time_to_diff_y=id_time_to_diff_y,
        valid_times=valid_times,
        formula=formula,
        exog_idx=exog_idx,
        transform=transform,
    )
end
