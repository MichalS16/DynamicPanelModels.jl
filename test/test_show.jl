# test/test_show.jl

# libraries
using Test
using DynamicPanelModels
using LinearAlgebra
using SparseArrays
using Distributions

@testset "Show" begin
    # Create a mock DynamicPanelResult
    n_obs = 15
    n_groups = 5
    n_inst = 10
    names = ["L.very_long_variable_name_exceeding_30_chars", "x1"]
    coef = [0.5, -0.2]
    vcov = Matrix(0.0001 * I(2))
    y = rand(n_obs)
    X = rand(n_obs, 2)
    Z = spzeros(n_obs, n_inst)
    residuals = randn(n_obs)
    j_stat = 5.0
    j_pval = 0.20
    fitted = y .- residuals

    # Metadata dictionary
    metadata = Dict(:method => "Two-Step System GMM", :formula => "y ~ lag(y) + x1")

    # Construct the object
    model = DynamicPanelResult(
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
        Matrix{Float64}(I, n_inst, n_inst),
        j_stat,
        j_pval;
        windmeijer=true,
        metadata=Dict{Symbol,Any}(:method => "Two-Step System GMM", :formula => "y ~ lag(y) + x1"),
    )

    # Significance stars test
    @testset "Significance Stars Logic" begin
        # Test various p-values
        @test DynamicPanelModels.significance_stars(0.0001) == "***"
        @test DynamicPanelModels.significance_stars(0.005) == "**"
        @test DynamicPanelModels.significance_stars(0.03) == "*"
        @test DynamicPanelModels.significance_stars(0.08) == "."
        @test DynamicPanelModels.significance_stars(0.20) == ""
        @test DynamicPanelModels.significance_stars(NaN) == ""
    end

    # Output content test
    @testset "Output Content" begin
        # Capture the output of show
        io = IOBuffer()
        show(io, model)
        output = String(take!(io))

        # Check Header and Metadata integration
        @test occursin("Dynamic Panel Data Estimation", output)
        @test occursin(r"Method:\s+Two-Step System GMM", output)
        @test occursin(r"Std. Errors:\s+Windmeijer \(2005\) corrected", output)

        # Check Table Headers
        @test occursin("Variable", output)
        @test occursin("Estimate", output)

        # Data & Stars
        @test occursin("0.50000", output)
        @test occursin("-0.20000", output)
        @test occursin("***", output)

        # Check truncation
        @test occursin("...", output)
        @test length(filter(c -> c == '.', output)) >= 3

        # Check Sargan Test
        @test occursin("Sargan/Hansen test", output)
        @test occursin("chi2(8) = 5.0000", output)
        @test occursin("(p = 0.2000)", output)
        @test occursin("instruments are valid; not rejected", output)

        # Check Footer
        @test occursin("For full diagnostics", output)
    end

    # Regression test: the "Std. Errors:" line must reflect `robust`/`windmeijer`
    # independently, not just `windmeijer` (which previously mislabeled
    # non-robust models as "Robust").
    @testset "Std. Errors label reflects robust/windmeijer independently" begin
        mk(; robust, windmeijer) = DynamicPanelResult(
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
            Matrix{Float64}(I, n_inst, n_inst),
            j_stat,
            j_pval;
            windmeijer=windmeijer,
            robust=robust,
            metadata=Dict{Symbol,Any}(:method => "Mock"),
        )

        io = IOBuffer()
        show(io, mk(; robust=false, windmeijer=false))
        @test occursin(r"Std. Errors:\s+Homoskedastic \(non-robust\)", String(take!(io)))

        io2 = IOBuffer()
        show(io2, mk(; robust=true, windmeijer=false))
        @test occursin(r"Std. Errors:\s+Robust \(cluster-sandwich\)", String(take!(io2)))

        io3 = IOBuffer()
        show(io3, mk(; robust=true, windmeijer=true))
        @test occursin(r"Std. Errors:\s+Windmeijer \(2005\) corrected", String(take!(io3)))
    end
end
