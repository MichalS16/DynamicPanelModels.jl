# test/test_instruments.jl

# libraries
using Test
using SparseArrays
using LinearAlgebra
using DynamicPanelModels

@testset "Instruments" begin
    # Create mock differenced data structure
    valid_times = [1, 2, 3, 4]
    id_time_to_y = Dict(
        1 => Dict(1=>10.0, 2=>20.0, 3=>30.0, 4=>40.0), 2 => Dict(1=>1.0, 2=>2.0, 3=>3.0, 4=>4.0)
    )

    # Differenced y values
    id_time_to_diff_y = Dict(
        1 => Dict(2=>10.0, 3=>10.0, 4=>10.0), 2 => Dict(2=>1.0, 3=>1.0, 4=>1.0)
    )

    # Panel info for differenced observations
    panel_info = [
        (id=1, time=3, is_level=false),
        (id=1, time=4, is_level=false),
        (id=2, time=3, is_level=false),
        (id=2, time=4, is_level=false),
        (id=1, time=3, is_level=true),
        (id=1, time=4, is_level=true),
        (id=2, time=3, is_level=true),
        (id=2, time=4, is_level=true),
    ]

    # Construct diff_data NamedTuple
    diff_data = (
        panel_info=panel_info,
        n_obs=length(panel_info),
        id_time_to_y=id_time_to_y,
        id_time_to_diff_y=id_time_to_diff_y,
        valid_times=valid_times,
    )

    # Difference GMM
    @testset "Difference GMM" begin
        model = DifferenceGMM()
        Z = build_instruments(model, diff_data)

        # Test dimensions and values
        @test size(Z, 2) == 3
        @test Z[1, 1] == 10.0
        @test Z[1, 2] == 0.0
        @test Z[2, 2] == 20.0
        @test Z[2, 3] == 10.0
        @test nnz(Z[5:8, :]) == 0
    end

    # Instrument lag-order limits (min_lag / max_lag)
    @testset "Difference GMM - lag limits" begin
        model = DifferenceGMM()
        # Default: t=3 uses lag order 2 (1 col), t=4 uses orders 2,3 (2 cols) -> 3
        @test size(build_instruments(model, diff_data), 2) == 3
        # min_lag=3 keeps only order >= 3: t=3 none, t=4 order 3 -> 1 col
        @test size(build_instruments(model, diff_data; min_lag=3), 2) == 1
        # max_lag=2 keeps only order <= 2: one col per differenced period -> 2
        @test size(build_instruments(model, diff_data; max_lag=2), 2) == 2

        # Inverted/degenerate bounds must error immediately rather than silently
        # producing a 0-column instrument matrix that later surfaces as a confusing
        # generic "under-identified" error from `estimate`.
        @test_throws ErrorException build_instruments(model, diff_data; min_lag=5, max_lag=3)
        @test_throws ErrorException build_instruments(model, diff_data; max_lags=0)
    end

    # Difference GMM with collapsed instruments
    @testset "Difference GMM - Collapse" begin
        # Collapsed instruments
        model = DifferenceGMM()
        Z_collapsed = build_instruments(model, diff_data; collapse=true)
        @test size(Z_collapsed, 2) == 2
        @test Z_collapsed[2, 1] == 20.0
        @test Z_collapsed[2, 2] == 10.0

        # Limited lags
        Z_limited = build_instruments(model, diff_data; max_lags=1)
        @test size(Z_limited, 2) == 2
    end

    # System GMM
    @testset "System GMM" begin
        # Full System GMM
        model = SystemGMM()
        Z = build_instruments(model, diff_data)

        # Check dimensions and values
        @test size(Z, 2) == 5
        @test Z[5, 4] == 10.0
        @test Z[6, 5] == 10.0
        @test nnz(Z[5:8, 1:3]) == 0
    end

    # Anderson-Hsiao
    @testset "Anderson-Hsiao" begin
        # Single instrument
        model = AndersonHsiao()
        Z = build_instruments(model, diff_data)
        @test size(Z, 2) == 1
        @test Z[1, 1] == 10.0
        @test Z[2, 1] == 20.0
        @test Z[3, 1] == 1.0
    end

    # Short panel: fewer than 3 periods yields no lag instruments
    @testset "Short panel (T < 3)" begin
        short = (
            panel_info=[(id=1, time=1, is_level=false), (id=1, time=2, is_level=false)],
            n_obs=2,
            id_time_to_y=Dict(1 => Dict(1 => 5.0, 2 => 6.0)),
            id_time_to_diff_y=Dict(1 => Dict(2 => 1.0)),
            valid_times=[1, 2],
        )
        @test size(build_instruments(DifferenceGMM(), short), 2) == 0
        # System GMM falls back to the (empty) difference block
        @test size(build_instruments(SystemGMM(), short), 2) == 0
    end

    # Exogenous regressors appended as IV-style instruments
    @testset "Exogenous regressor instrumenting" begin
        # X column 1 aligned with panel_info rows: differenced rows get nonzero
        # values, level rows get different values, to check the block-diagonal
        # (is_level-respecting) placement.
        X = reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 8, 1)
        diff_data_exog = merge(diff_data, (X=X, exog_idx=[1]))

        @testset "DifferenceGMM" begin
            model = DifferenceGMM()
            Z_no = build_instruments(DifferenceGMM(), diff_data)
            Z = build_instruments(model, diff_data_exog)
            @test size(Z, 2) == size(Z_no, 2) + 1
            # Differenced rows (1:4) get their own X value in the new last column
            @test Z[1:4, end] == X[1:4, 1]
            # Level rows (5:8) get zero in the differenced-equation instrument block
            @test all(Z[5:8, end] .== 0.0)
        end

        @testset "SystemGMM" begin
            model = SystemGMM()
            Z_no = build_instruments(SystemGMM(), diff_data)
            Z = build_instruments(model, diff_data_exog)
            # One exog column appended to each of the two blocks (diff + level)
            @test size(Z, 2) == size(Z_no, 2) + 2
        end

        @testset "AndersonHsiao" begin
            model = AndersonHsiao()
            Z_no = build_instruments(AndersonHsiao(), diff_data)
            Z = build_instruments(model, diff_data_exog)
            @test size(Z, 2) == size(Z_no, 2) + 1
            @test Z[1:4, end] == X[1:4, 1]
        end
    end

    # A model type with no build_instruments specialization must raise an
    # informative error (naming the offending type), not a generic MethodError.
    struct _UnimplementedModel <: AbstractDynamicPanelModel end
    @testset "Unimplemented model falls back to an informative error" begin
        result = @test_throws ErrorException build_instruments(_UnimplementedModel(), diff_data)
        @test occursin("not implemented", result.value.msg)
        @test occursin("_UnimplementedModel", result.value.msg)
    end
end
