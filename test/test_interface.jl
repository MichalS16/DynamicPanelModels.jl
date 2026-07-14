# test/test_interface.jl

# libraries
using Test
using DynamicPanelModels
using StatsAPI
using StatsBase
using LinearAlgebra
using SparseArrays
using Distributions
using DataFrames
using Random

@testset "Interface" begin
    # Create a mock DynamicPanelResult
    n_obs = 15
    n_groups = 5
    n_inst = 10
    names = ["L.y", "x1"]
    y = rand(n_obs)
    X = rand(n_obs, 2)
    Z = spzeros(n_obs, n_inst)
    residuals_vec = randn(n_obs)
    fitted_vec = y .- residuals_vec
    coef_vec = [0.5, -0.2]
    vcov_mat = Matrix(0.04 * I(2))
    j_stat = 5.0
    j_pval = 0.20

    # Construct the object
    model = DynamicPanelResult(
        coef_vec,
        vcov_mat,
        residuals_vec,
        fitted_vec,
        n_obs,
        n_groups,
        10,
        ["L.y", "x1"],
        X,
        y,
        Z,
        Matrix{Float64}(I, 10, 10),
        j_stat,
        j_pval;
        windmeijer=true,
        metadata=Dict{Symbol,Any}(
            :method => "Mock",
            :formula => "y ~ lag(y) + x1",
            :ar_tests => Dict(
                1 => DynamicPanelTest("Arellano-Bond AR(1)", 1.5, 0, 0.13),
                2 => DynamicPanelTest("Arellano-Bond AR(2)", 0.4, 0, 0.69),
            ),
        ),
    )

    # Basic Accessors
    @testset "Basic Accessors" begin
        @test coef(model) === model.coef
        @test vcov(model) === model.vcov
        @test nobs(model) == n_obs
        @test dof(model) == 2
        @test formula(model) == "y ~ lag(y) + x1"
    end

    # Inference and Robustness
    @testset "Inference & Robustness" begin
        # R-squared and Robustness
        @test r2(model) isa Float64
        @test is_robust(model) == true

        # Confidence Intervals
        ci = confint(model; level=0.95)
        @test size(ci) == (2, 2)

        # Coefficient Table
        ct = coeftable(model)
        @test ct isa CoefTable
    end

    # Diagnostics Accessors
    @testset "Diagnostic Accessors" begin
        # Test Sargan and AR tests
        sargan = sargan_test(model)
        @test sargan isa DynamicPanelTest
        @test sargan.test_name == "Sargan J-Test"
        @test ar_test(model, 1).stat == 1.5
        @test ar_test(model, 2).stat == 0.4

        # Requesting an order that was never computed raises a clear error
        @test_throws ErrorException ar_test(model, 3)
    end

    # Package Specific Accessors
    @testset "Package Specifics" begin
        @test ngroups(model) == n_groups
    end

    # Regression test: fit() must honor the model spec's steps/robust fields,
    # not silently fall back to estimate()'s own keyword defaults.
    @testset "fit() honors model spec (steps/robust)" begin
        Random.seed!(2024)
        N, T = 150, 6
        rows = NamedTuple[]
        for i in 1:N
            eta = randn()
            y_prev = randn()
            for t in 1:T
                x = randn()
                y = 0.5 * y_prev + 0.8 * x + eta + 0.5 * randn()
                push!(rows, (id=i, t=t, y=y, x=x))
                y_prev = y
            end
        end
        df = DataFrame(rows)

        m1 = fit(
            DifferenceGMM(robust=true, steps=1),
            df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
        )
        m2 = fit(
            DifferenceGMM(robust=true, steps=2),
            df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
        )

        # Two-step estimates must differ from one-step (re-weighting occurred)
        @test coef(m1) != coef(m2)
        @test occursin("1-step", m1.metadata[:method])
        @test occursin("2-step", m2.metadata[:method])
        @test m2.windmeijer == true

        # robust=false must skip the Windmeijer correction
        m3 = fit(
            DifferenceGMM(robust=false, steps=2),
            df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
        )
        @test m3.windmeijer == false

        # Explicit kwargs to fit() still override the model spec
        m4 = fit(
            DifferenceGMM(robust=true, steps=1),
            df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            steps=2,
        )
        @test occursin("2-step", m4.metadata[:method])
    end
end
