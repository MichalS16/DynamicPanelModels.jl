# test/test_types.jl

# libraries
using Test
using DynamicPanelModels
using SparseArrays
using LinearAlgebra
using StatsAPI: RegressionModel

@testset "Types" begin
    # Test Abstract Type Hierarchy
    @testset "Abstract Hierarchy" begin
        @test isabstracttype(AbstractDynamicPanelModel)
        @test DifferenceGMM <: AbstractDynamicPanelModel
        @test SystemGMM <: AbstractDynamicPanelModel
        @test AndersonHsiao <: AbstractDynamicPanelModel
    end

    # Test Estimator Structs
    @testset "Estimator Structs" begin
        # DifferenceGMM
        diff_model = DifferenceGMM()
        @test diff_model.robust == false
        @test diff_model.steps == 1
        @test diff_model.windmeijer == true

        # Test custom constructor
        diff_custom = DifferenceGMM(robust=true, steps=2, windmeijer=false)
        @test diff_custom.robust == true
        @test diff_custom.steps == 2
        @test diff_custom.windmeijer == false

        # SystemGMM
        sys_model = SystemGMM(steps=2)
        @test sys_model.steps == 2
        @test_throws ArgumentError SystemGMM(steps=0)

        # AndersonHsiao
        @test AndersonHsiao() isa AbstractDynamicPanelModel
    end

    # Test Result Structures
    @testset "DynamicPanelResult Structure" begin
        # Create synthetic data for testing
        coef = [0.5, 0.2]
        vcov = [0.1 0.0; 0.0 0.1]
        resid = rand(10)
        fitted = rand(10)
        X = rand(10, 2)
        y = rand(10)
        Z = spzeros(10, 5)
        W = rand(5, 5)

        # Construct DynamicPanelResult
        result = DynamicPanelResult(
            coef,
            vcov,
            resid,
            fitted,
            10,
            5,
            5,
            ["L.y", "x"],
            X,
            y,
            Z,
            W,
            0.05,
            0.80;
            windmeijer=true,
            metadata=Dict{Symbol,Any}(:test => "data"),
        )

        # Check type hierarchy
        @test result isa RegressionModel
        @test result isa DynamicPanelResult{Float64,Matrix{Float64},SparseMatrixCSC{Float64,Int64}}
        @test issparse(result.Z)

        # Check field shapes and types
        @test result.windmeijer == true
        @test result.metadata[:test] == "data"
        @test result.n_obs == 10

        # `robust` defaults to `windmeijer` when not given explicitly, but can
        # be set independently (e.g. steps=2, robust=true, windmeijer=false).
        @test result.robust == true  # defaulted from windmeijer=true above
        result_no_default = DynamicPanelResult(
            coef,
            vcov,
            resid,
            fitted,
            10,
            5,
            5,
            ["L.y", "x"],
            X,
            y,
            Z,
            W,
            0.05,
            0.80;
            windmeijer=false,
            robust=true,
            metadata=Dict{Symbol,Any}(),
        )
        @test result_no_default.windmeijer == false
        @test result_no_default.robust == true

        # Test dimension mismatch error
        bad_coef = [0.5, 0.2, 0.1]
        @test_throws DimensionMismatch DynamicPanelResult(
            bad_coef, vcov, resid, fitted, 10, 5, 5, ["L.y", "x1", "x2"], X, y, Z, W, 0.05, 0.80
        )
    end

    # Test diagnostic structure
    @testset "DynamicPanelTest Structure" begin
        t = DynamicPanelTest("AR(2)", 1.96, 1, 0.05)
        @test t.test_name == "AR(2)"
        @test t.stat == 1.96
        @test t.pvalue == 0.05
    end
end
