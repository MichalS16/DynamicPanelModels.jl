# test/test_plot_recipes.jl

# libraries
using Test
using DynamicPanelModels
using RecipesBase
using LinearAlgebra
using SparseArrays
using Statistics
using Distributions: Normal, quantile

@testset "Plot" begin
    # Create a mock DynamicPanelResult
    n_obs = 15
    n_groups = 5
    y = rand(n_obs)
    X = rand(n_obs, 2)
    Z = spzeros(n_obs, 10)
    residuals = randn(n_obs)
    fitted = y .- residuals
    coef = [0.5, -0.2]
    vcov = Matrix(0.01 * I(2))
    j_stat = 5.0
    j_pval = 0.20

    # Construct the object
    model = DynamicPanelResult(
        coef,
        vcov,
        residuals,
        fitted,
        n_obs,
        n_groups,
        10,
        ["L.y", "x1"],
        X,
        y,
        Z,
        Matrix{Float64}(I, 10, 10),
        5.0,
        0.20;
        windmeijer=false,
        metadata=Dict{Symbol,Any}(:method => "Mock", :formula => "y ~ L.y + x"),
    )

    # Dashboard Layout
    @testset "Dashboard Layout" begin
        # Dashboard Layout Recipe
        recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), model, Val(:dashboard))
        @test length(recipes) == 4
        @test recipes[1].plotattributes[:title] == "GMM Diagnostics (N=5, T*=15)"
    end

    # Standardized Residuals Recipe
    @testset "Standardized Residuals Plot" begin
        # Test length and content of recipe
        recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), model, Val(:residuals))
        @test length(recipes) == 3

        # Check standardized residuals
        scatter_idx = findfirst(
            s -> get(s.plotattributes, :seriestype, nothing) == :scatter, recipes
        )
        res_data = recipes[scatter_idx].args[1]
        @test isapprox(mean(res_data), 0.0, atol=1e-10)
        @test isapprox(std(res_data), 1.0, atol=1e-10)
        @test recipes[scatter_idx].plotattributes[:title] == "Standardized Residuals"
    end

    # Residuals Recipe
    @testset "Fitted vs Actual Plot" begin
        # Fitted vs Actual Recipe
        recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), model, Val(:fitted))
        @test length(recipes) == 2

        # Find the identity line series
        id_line_idx = findfirst(s -> get(s.plotattributes, :label, "") == "Identity", recipes)
        @test recipes[id_line_idx].plotattributes[:seriestype] == :path
        @test recipes[id_line_idx].plotattributes[:linestyle] == :dash
    end

    # Histogram Recipe
    @testset "Histogram Plot" begin
        # Histogram Recipe
        recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), model, Val(:histogram))
        @test length(recipes) == 2

        # Find the histogram series
        hist_idx = findfirst(
            s -> get(s.plotattributes, :seriestype, nothing) == :histogram, recipes
        )
        @test recipes[hist_idx].plotattributes[:normalize] == true
        @test recipes[hist_idx].plotattributes[:fillcolor] == :purple
    end

    # Q-Q Plot Recipe
    @testset "Q-Q Plot" begin
        # Test Q-Q Plot Recipe
        recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), model, Val(:qq))
        @test length(recipes) == 2

        # Find the Q-Q plot series
        qq_idx = findfirst(s -> get(s.plotattributes, :title, "") == "Normal Q-Q Plot", recipes)
        @test !isnothing(qq_idx)

        # Check that the points fall approximately on the y=x line
        qq_scatter = recipes[findfirst(s -> s.plotattributes[:seriestype] == :scatter, recipes)]
        @test isapprox(std(qq_scatter.args[2]), 1.0, atol=1e-10)

        # Regression guard: theoretical and sample quantiles must be the same
        # length (one point per observation) and follow the standard Blom
        # plotting positions (i - 0.5)/n, i = 1:n -- a plausible-looking but
        # wrong range expression previously produced n-1 theoretical
        # quantiles at positions i/n, silently misaligning every point.
        theo_q, sample_q = qq_scatter.args[1], qq_scatter.args[2]
        @test length(theo_q) == length(sample_q) == n_obs
        expected_probs = ((1:n_obs) .- 0.5) ./ n_obs
        @test isapprox(theo_q, quantile.(Normal(), expected_probs))

        # Sample quantiles must actually be sorted ascending (matching the
        # ascending theoretical quantiles) -- std==1 alone wouldn't catch an
        # unsorted/shuffled scatter with the same overall spread.
        @test issorted(qq_scatter.args[2])
    end

    # Degenerate residuals: zero variance (all-equal or n=1) previously fed
    # NaN (0/0 from StatsBase.zscore's std=0 denominator) into Plots'
    # internal histogram/QQ binning, crashing with an opaque ArgumentError
    # three frames deep in Base.reduce rather than rendering or erroring
    # clearly (only reproducible with Plots loaded, so this checks the
    # recipe-level fix directly: no NaN reaches the series data).
    @testset "Degenerate residuals: zero variance / single observation" begin
        @test DynamicPanelModels._safe_zscore([0.3, 0.3, 0.3]) == zeros(3)
        @test DynamicPanelModels._safe_zscore([0.5]) == zeros(1)
        @test DynamicPanelModels._safe_zscore(randn(10)) != zeros(10)  # normal case unaffected

        mk(resid) = DynamicPanelResult(
            coef,
            vcov,
            resid,
            resid .- resid,
            length(resid),
            n_groups,
            10,
            ["L.y", "x1"],
            rand(length(resid), 2),
            rand(length(resid)),
            spzeros(length(resid), 10),
            Matrix{Float64}(I, 10, 10),
            5.0,
            0.20;
            windmeijer=false,
            metadata=Dict{Symbol,Any}(:method => "Mock"),
        )

        for pt in (:residuals, :histogram, :qq, :dashboard)
            for m in (mk(fill(0.3, 10)), mk([0.5]))
                recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), m, Val(pt))
                vals = Iterators.flatten(
                    a for s in recipes for a in s.args if a isa AbstractVector{<:Real}
                )
                @test !any(v -> isnan(v) || isinf(v), vals)
            end
        end
    end

    # All-NaN/Inf residuals (e.g. a hand-built or catastrophically failed
    # result) must raise a clear error from the recipe itself, not propagate
    # an empty series into Plots' binning.
    @testset "All-non-finite residuals raise a clear error" begin
        bad = DynamicPanelResult(
            coef,
            vcov,
            [NaN, Inf, -Inf],
            zeros(3),
            3,
            n_groups,
            10,
            ["L.y", "x1"],
            rand(3, 2),
            rand(3),
            spzeros(3, 10),
            Matrix{Float64}(I, 10, 10),
            5.0,
            0.20;
            windmeijer=false,
            metadata=Dict{Symbol,Any}(:method => "Mock"),
        )
        for pt in (:histogram, :qq)
            result = @test_throws ErrorException RecipesBase.apply_recipe(
                Dict{Symbol,Any}(), bad, Val(pt)
            )
            @test occursin("No finite residuals", result.value.msg)
        end
    end

    # Error Handling Recipe
    @testset "Invalid Plot Type" begin
        @test_throws ErrorException RecipesBase.apply_recipe(
            Dict{Symbol,Any}(), model, Val(:non_existent)
        )
    end
end
