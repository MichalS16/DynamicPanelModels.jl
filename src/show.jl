# src/show.jl

"""
    significance_stars(p::Real)

Return significance stars for a p-value following standard econometric conventions.
`NaN` (e.g. an undefined test) maps to an empty string.

# Arguments
- `p::Real`: P-value from a statistical test.

# Returns
- `String`: Significance stars (`"***"`, `"**"`, `"*"`, `"."`, or `""`) for table or regression reporting.
"""
function significance_stars(p::Real)
    isnan(p) && return ""
    p < 0.001 && return "***"
    p < 0.01 && return "**"
    p < 0.05 && return "*"
    p < 0.1 && return "."
    return ""
end

"""
    _sargan_hint(j_pval::Real)

Interpretation hint for a Sargan/Hansen overidentification test at the 5% level.
"""
function _sargan_hint(j_pval::Real)
    return if j_pval > 0.05
        "instruments are valid; not rejected"
    else
        "H0 rejected; instruments may be invalid"
    end
end

"""
    Base.show(io::IO, model::DynamicPanelResult)

Display a `DynamicPanelResult` in a concise Stata-like summary table with adaptive column widths.

# Arguments
- `io::IO`: Output stream (e.g., `stdout`).
- `model::DynamicPanelResult`: Panel estimation results.

# Output
- Model header: method, formula, standard error type.
- Observations, groups, and instruments summary.
- Coefficient table with Estimate, Std. Error, z-value, p-value, and significance stars.
- Hansen/Sargan J-test for overidentifying restrictions (with interpretation hint).
- Guidance for additional diagnostics.
"""
function Base.show(io::IO, model::DynamicPanelResult)
    # Extract coefficient table
    ct = coeftable(model)
    names = ct.rownms
    max_name_len = maximum(length, names)
    col_width = clamp(max_name_len + 2, 12, 30)
    line_width = max(78, col_width + 55)
    sep = "─" ^ line_width
    double_sep = "═" ^ line_width

    # Extract method and formula from metadata if available
    method_str = get(model.metadata, :method, "One-Step GMM")
    formula_str = get(model.metadata, :formula, "N/A")

    # print Model Summary
    println(io)
    println(io, double_sep)
    println(io, "  Dynamic Panel Data Estimation")
    println(io, double_sep)
    @printf(io, "  %-22s %-s\n", "Method:", method_str)
    @printf(io, "  %-22s %-s\n", "Formula:", formula_str)
    se_type = if model.windmeijer
        "Windmeijer (2005) corrected"
    elseif model.robust
        "Robust (cluster-sandwich)"
    else
        "Homoskedastic (non-robust)"
    end
    @printf(io, "  %-22s %-s\n", "Std. Errors:", se_type)
    println(io, sep)
    @printf(
        io, "  %-22s %12d    %-15s %12d\n", "Observations:", model.n_obs, "Groups:", model.n_groups
    )
    @printf(io, "  %-22s %12d\n", "Instruments:", model.n_instruments)
    println(io, sep)
    print(io, "  ", rpad("Variable", col_width))
    @printf(io, " %12s %12s %10s %10s\n", "Estimate", "Std. Error", "z-value", "Pr(>|z|)")
    println(io, sep)

    # Coefficient Table
    for i in eachindex(names)
        # Extract values
        est = ct.cols[1][i]
        se = ct.cols[2][i]
        z = ct.cols[3][i]
        p = ct.cols[4][i]
        stars = significance_stars(p)

        # Truncate long variable names if necessary
        name_display = names[i]
        if length(name_display) > col_width
            name_display = name_display[1:(col_width - 3)] * "..."
        end

        # Print results
        print(io, "  ", rpad(name_display, col_width))
        @printf(io, " %12.5f %12.5f %10.4f %10.4f %-3s\n", est, se, z, p, stars)
    end

    # Footer Section
    println(io, sep)
    println(io, "  Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
    println(io)

    # Sargan/Hansen Test Section
    if !isnan(model.j_stat)
        dof_j = model.n_instruments - length(names)
        println(io, "  Sargan/Hansen test of overidentifying restrictions:")
        @printf(io, "    chi2(%d) = %.4f (p = %.4f)\n", dof_j, model.j_stat, model.j_pval)

        # Interpretation Hint
        println(io, "    (H0: $(_sargan_hint(model.j_pval)) at 5%)")
    end

    # Diagnostics Hint
    println(io)
    println(io, "  For full diagnostics (AR tests, etc.), run: diagnose(model)")
    return println(io, double_sep)
end

# Support for plain text MIME type
Base.show(io::IO, ::MIME"text/plain", model::DynamicPanelResult) = show(io, model)
