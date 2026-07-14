# test/test_estimation.jl

# libraries
using Test
using LinearAlgebra
using SparseArrays
using DataFrames
using DynamicPanelModels

@testset "Estimation" begin
    # Set up mock diff_data
    X_mat = reshape([1.0, 2.0, 0.5, 1.5], 4, 1)
    y = [0.55, 0.95, 0.20, 0.80]
    coef_names = ["L.y"]
    valid_times = [1, 2, 3, 4]

    # Instrument Mapping
    id_time_to_y = Dict(
        1 => Dict(1=>10.0, 2=>20.0, 3=>30.0, 4=>40.0), 2 => Dict(1=>1.0, 2=>2.0, 3=>3.0, 4=>4.0)
    )
    id_time_to_diff_y = Dict(
        1 => Dict(2=>10.0, 3=>10.0, 4=>10.0), 2 => Dict(2=>1.0, 3=>1.0, 4=>1.0)
    )

    # Panel Info
    panel_info = [
        (id=1, time=3, is_level=false),
        (id=1, time=4, is_level=false),
        (id=2, time=3, is_level=false),
        (id=2, time=4, is_level=false),
    ]

    # Assemble diff_data
    diff_data = (
        y=y,
        X=X_mat,
        coef_names=coef_names,
        panel_info=panel_info,
        n_obs=4,
        n_groups=2,
        id_time_to_y=id_time_to_y,
        id_time_to_diff_y=id_time_to_diff_y,
        valid_times=valid_times,
        formula="y ~ L.y",
    )

    # One-Step Difference GMM
    @testset "One-Step GMM" begin
        # Basic One-Step GMM
        model = DifferenceGMM()
        result = estimate(model, diff_data; steps=1, robust=true)
        @test result isa DynamicPanelResult
        @test length(result.coef) == 1
        @test !result.windmeijer
        @test isapprox(result.coef[1], 0.5, atol=0.1)
        @test occursin("1-step", result.metadata[:method])
    end

    # One-Step non-robust (homoskedastic) standard errors
    @testset "One-Step homoskedastic SEs" begin
        result = estimate(DifferenceGMM(), diff_data; steps=1, robust=false)
        @test !result.windmeijer
        @test all(diag(result.vcov) .>= 0)
        # Non-robust one-step differs from clustered-robust one-step in general
        result_rob = estimate(DifferenceGMM(), diff_data; steps=1, robust=true)
        @test result.vcov != result_rob.vcov
    end

    # Two-Step GMM with Windmeijer Correction
    @testset "Two-Step & Windmeijer" begin
        # Basic Two-Step GMM
        model = DifferenceGMM()
        res_no_w = estimate(model, diff_data; steps=2, robust=false)
        res_w = estimate(model, diff_data; steps=2, robust=true)
        @test res_w.windmeijer == true
        @test occursin("2-step", res_w.metadata[:method])
        @test stderror(res_w)[1] >= stderror(res_no_w)[1]
    end

    # Under-identification & Small Sample
    @testset "Identification Guards" begin
        # Under-identified: n_obs (4) >= regressors (4) but instruments (3) < regressors
        X_wide = rand(4, 4)
        diff_underid = merge(diff_data, (X=X_wide, coef_names=["x$i" for i in 1:4]))
        err_underid = try
            estimate(DifferenceGMM(), diff_underid)
            nothing
        catch e
            e
        end
        @test err_underid isa ErrorException
        @test occursin("under-identified", err_underid.msg)

        # Too few observations (n_obs < regressors)
        X_small = rand(2, 3)
        y_small = [1.0, 2.0]
        diff_small = merge(
            diff_data,
            (
                X=X_small,
                y=y_small,
                n_obs=2,
                panel_info=panel_info[1:2],
                coef_names=["x1", "x2", "x3"],
            ),
        )
        err_small = try
            estimate(DifferenceGMM(), diff_small)
            nothing
        catch e
            e
        end
        @test err_small isa ErrorException
        @test occursin("Insufficient observations", err_small.msg)
    end

    # System GMM Mechanics
    @testset "System GMM Execution" begin
        # Create System GMM data
        y_sys = [y; y]
        X_sys = [X_mat; X_mat]
        panel_info_sys = [
            panel_info;
            [
                (id=1, time=3, is_level=true),
                (id=1, time=4, is_level=true),
                (id=2, time=3, is_level=true),
                (id=2, time=4, is_level=true),
            ]
        ]

        # Assemble
        diff_data_sys = merge(diff_data, (y=y_sys, X=X_sys, panel_info=panel_info_sys, n_obs=8))

        # Estimate System GMM. One instrument column is collinear in this tiny
        # mock, so the default (drop_collinear=true) reports 4; disabling it keeps 5.
        result = estimate(SystemGMM(), diff_data_sys; steps=2)
        @test result.n_instruments == 4
        @test length(result.coef) == 1
        result_keep = estimate(SystemGMM(), diff_data_sys; steps=2, drop_collinear=false)
        @test result_keep.n_instruments == 5
    end
end
