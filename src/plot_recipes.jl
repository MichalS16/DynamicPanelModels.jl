# src/plot_recipes.jl

"""
    plot_recipe(model::DynamicPanelResult; plot_type::Symbol=:dashboard)

Plots a `DynamicPanelResult` using RecipesBase/Plots.jl.

# Arguments
- `model::DynamicPanelResult`: The model object to plot.
- `plot_type::Symbol = :dashboard`: Type of plot. Options: `:dashboard`, `:residuals`, `:fitted`, `:histogram`, `:qq`.

# Returns
- `(model, Val(plot_type))` for dispatching to the corresponding plot recipe.
"""
@recipe function f(model::DynamicPanelResult, plot_type::Symbol=:dashboard)
    return model, Val(plot_type)
end

"""
    plot_recipe(model::DynamicPanelResult, ::Val{:dashboard})

Generate a 4-panel diagnostic dashboard for a `DynamicPanelResult`.

# Panels
1. Standardized residuals vs. observation index
2. Fitted vs. actual values
3. Histogram of standardized residuals
4. Q–Q plot of standardized residuals

# Arguments
- `model::DynamicPanelResult`: Model estimation results with residuals, fitted values, and other outputs.

# Returns
- A recipe tuple compatible with RecipesBase/Plots.jl.
"""
@recipe function f(model::DynamicPanelResult, ::Val{:dashboard})
    # Dashboard layout
    title := "GMM Diagnostics (N=$(model.n_groups), T*=$(model.n_obs))"
    layout := (2, 2)
    size := (900, 700)
    legend := false

    @series begin
        subplot := 1
        (model, :residuals)
    end

    @series begin
        subplot := 2
        (model, :fitted)
    end

    @series begin
        subplot := 3
        (model, :histogram)
    end

    @series begin
        subplot := 4
        (model, :qq)
    end
end

"""
    plot_recipe(model::DynamicPanelResult, ::Val{:residuals})

Create a residual diagnostics plot for a `DynamicPanelResult`, showing standardized
residuals by observation index.

# Arguments
- `model::DynamicPanelResult`: Estimation results containing residuals.

# Returns
- A recipe tuple compatible with RecipesBase/Plots.jl.
"""
@recipe function f(model::DynamicPanelResult, ::Val{:residuals})
    # Residuals Plot
    resid = model.residuals
    std_resid = (resid .- mean(resid)) ./ std(resid)

    title --> "Standardized Residuals"
    xguide --> "Observation Index"
    yguide --> "Std. Residuals"
    grid --> true

    # Zero reference line
    @series begin
        seriestype := :hline
        linecolor := :black
        linewidth := 1.5
        label := ""
        [0.0]
    end

    # Threshold lines for outliers
    @series begin
        seriestype := :hline
        linecolor := :red
        linestyle := :dot
        label := "±2σ"
        [2.0, -2.0]
    end

    # Data points
    @series begin
        seriestype := :scatter
        markercolor --> :blue
        markeralpha --> 0.6
        markerstrokewidth --> 0
        label := "Residuals"
        std_resid
    end
end

"""
    plot_recipe(model::DynamicPanelResult, ::Val{:fitted})

Create a scatter plot of fitted vs. actual values from a `DynamicPanelResult` 
to evaluate model fit.

# Arguments
- `model::DynamicPanelResult`: Estimation results containing `fitted` and `y` values.

# Returns
- `Recipe`: Tuple compatible with RecipesBase/Plots.jl for plotting.
"""
@recipe function f(model::DynamicPanelResult, ::Val{:fitted})
    # Fitted vs Actual Plot
    fit_vals = model.fitted
    act_vals = model.y

    # Plot Settings
    title --> "Fitted vs Actual Values"
    xguide --> "Fitted"
    yguide --> "Actual"
    aspect_ratio --> :equal

    # Scatter of fitted vs actual
    @series begin
        seriestype := :scatter
        markercolor --> :green
        markeralpha --> 0.5
        markerstrokewidth --> 0
        label := "Observations"
        fit_vals, act_vals
    end

    # 45-degree identity line
    if !isempty(fit_vals) && !isempty(act_vals)
        all_vals = vcat(fit_vals, act_vals)
        mn, mx = extrema(filter(isfinite, all_vals))
        @series begin
            seriestype := :path
            linestyle := :dash
            linecolor := :red
            label := "Identity"
            [mn, mx], [mn, mx]
        end
    end
end

"""
    plot_recipe(model::DynamicPanelResult, ::Val{:histogram})

Generate a normalized histogram of standardized residuals from a `DynamicPanelResult` 
with an overlaid standard normal curve to assess residual distribution.

# Arguments
- `model::DynamicPanelResult`: Estimation results containing residuals.

# Returns
- Recipe tuple compatible with RecipesBase/Plots.jl.
"""
@recipe function f(model::DynamicPanelResult, ::Val{:histogram})
    # Data Preparation
    resid = filter(isfinite, model.residuals)
    std_resid = (resid .- mean(resid)) ./ std(resid)

    # Plot Settings
    title --> "Residual Density"
    xguide --> "Standardized Residual"
    yguide --> "Density"

    # Histogram of standardized residuals
    @series begin
        seriestype := :histogram
        normalize := true
        fillcolor --> :purple
        fillalpha --> 0.4
        label := "Empirical"
        std_resid
    end

    # Theoretical Normal Curve
    @series begin
        seriestype := :path
        linecolor := :black
        linewidth := 2
        linestyle := :dash
        label := "Normal(0,1)"
        x_grid = range(-4, 4, length=100)
        y_grid = pdf.(Normal(0, 1), x_grid)
        x_grid, y_grid
    end
end

"""
    plot_recipe(model::DynamicPanelResult, ::Val{:qq})

Generate a Q–Q plot of standardized residuals from a `DynamicPanelResult`
to assess normality.

# Arguments
- `model::DynamicPanelResult`: Estimation results containing residuals.

# Returns
- Recipe tuple for plotting with RecipesBase/Plots.jl.
"""
@recipe function f(model::DynamicPanelResult, ::Val{:qq})
    # Q-Q Plot
    raw_resid = filter(isfinite, collect(skipmissing(model.residuals)))
    n = length(raw_resid)

    # Standardized residuals and theoretical quantiles
    std_resid = (raw_resid .- mean(raw_resid)) ./ std(raw_resid)
    sorted_resid = sort(std_resid)
    probs = (1:(n .- 0.5)) ./ n
    theo_q = quantile.(Normal(), probs)

    # Plot Settings
    title --> "Normal Q-Q Plot"
    xguide --> "Theoretical Quantiles"
    yguide --> "Sample Quantiles"

    # Q-Q Scatter
    @series begin
        seriestype := :scatter
        markercolor --> :orange
        markeralpha --> 0.7
        markerstrokewidth --> 0
        label := "Residuals"
        theo_q, sorted_resid
    end

    # Reference line
    mn, mx = extrema(vcat(theo_q, sorted_resid))
    @series begin
        seriestype := :path
        linestyle := :dash
        linecolor := :red
        label := "Normal"
        [mn, mx], [mn, mx]
    end
end

"""
    plot_recipe(model::DynamicPanelResult, v::Val{T})

Fallback plotting recipe for unsupported plot types of a `DynamicPanelResult`.

# Arguments
- `model::DynamicPanelResult`: Estimation result object.
- `v::Val{T}`: Plot type identifier.
"""
@recipe function f(model::DynamicPanelResult, v)
    error("Unknown plot type. Use: :dashboard, :residuals, :fitted, :histogram, or :qq")
end
