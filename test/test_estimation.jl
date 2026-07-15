# test/test_estimation.jl

# libraries
using Test
using LinearAlgebra
using SparseArrays
using DataFrames
using DynamicPanelModels

@testset "Estimation" begin
    @testset "posdef_fix" begin
        # Already positive definite: returned unchanged
        A_pd = [2.0 0.0; 0.0 2.0]
        @test DynamicPanelModels.posdef_fix(A_pd) == A_pd

        # Mildly negative eigenvalue (floating-point-noise scale): shifted
        # just enough to become positive definite.
        A_mild = [1e-13 0.0; 0.0 1.0]
        fixed_mild = DynamicPanelModels.posdef_fix(A_mild)
        @test isposdef(Symmetric(fixed_mild))

        # Severely rank-deficient input: still fixed to positive definite.
        A_bad = [-1.0 0.0; 0.0 1.0]
        @test isposdef(Symmetric(DynamicPanelModels.posdef_fix(A_bad)))
    end

    @testset "_lag_vector shared idx_map matches the rebuilt-per-call form" begin
        id = [1, 1, 1, 2, 2, 2]
        time = [1, 2, 3, 1, 2, 3]
        v = [10.0, 20.0, 30.0, 100.0, 200.0, 300.0]

        idx_map = DynamicPanelModels._id_time_index(id, time)
        @test DynamicPanelModels._lag_vector(v, id, time, 1, idx_map) ==
            DynamicPanelModels._lag_vector(v, id, time, 1)
    end

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

        # Wiring guard (not an independent formula check — this calls the same
        # production calculate_windmeijer_correction that _solve_gmm calls, so
        # it cannot catch a bug in that function's math; it only catches
        # _solve_gmm assembling the wrong arguments/order/stale values around
        # it). Independent formula correctness is covered by
        # test_integration.jl's "Windmeijer correction sanity" test against a
        # simulated DGP with a known answer.
        Z = DynamicPanelModels._drop_collinear_columns(build_instruments(model, diff_data))
        ZtX_check = Z' * X_mat
        Zty_check = Z' * y
        W1_check = DynamicPanelModels.initial_weight_matrix(model, Z, diff_data)
        bread1_check = inv(DynamicPanelModels.posdef_fix(ZtX_check' * W1_check * ZtX_check))
        β1_check = bread1_check * (ZtX_check' * W1_check * Zty_check)
        res1_check = y - X_mat * β1_check
        Ω_check = DynamicPanelModels.calculate_clustered_weight_matrix(Z, res1_check, panel_info, 2)
        W2_check = inv(DynamicPanelModels.posdef_fix(Ω_check))
        bread2_check = inv(DynamicPanelModels.posdef_fix(ZtX_check' * W2_check * ZtX_check))
        β2_check = bread2_check * (ZtX_check' * W2_check * Zty_check)
        V1r_check =
            bread1_check * (ZtX_check' * W1_check * Ω_check * W1_check * ZtX_check) * bread1_check
        V_corr_check = DynamicPanelModels.calculate_windmeijer_correction(
            Z,
            X_mat,
            res1_check,
            β2_check,
            ZtX_check,
            Zty_check,
            W2_check,
            bread2_check,
            V1r_check,
            panel_info,
        )
        @test isapprox(V_corr_check, res_w.vcov)

        # Guard that calculate_windmeijer_correction's `ranges` kwarg is
        # actually consumed: a deliberately different partition (one big
        # cluster instead of two, changing which residuals/regressors are
        # treated as co-clustered) must change the corrected variance.
        V_corr_wrong_ranges = DynamicPanelModels.calculate_windmeijer_correction(
            Z,
            X_mat,
            res1_check,
            β2_check,
            ZtX_check,
            Zty_check,
            W2_check,
            bread2_check,
            V1r_check,
            panel_info;
            ranges=[1:4],
        )
        @test !isapprox(V_corr_wrong_ranges, V_corr_check)
    end

    # Under-identification & Small Sample
    @testset "Identification Guards" begin
        # Under-identified: n_obs (4) >= regressors (4) but instruments (3) < regressors
        X_wide = rand(4, 4)
        diff_underid = merge(diff_data, (X=X_wide, coef_names=["x$i" for i in 1:4]))
        result_underid = @test_throws ErrorException estimate(DifferenceGMM(), diff_underid)
        @test occursin("under-identified", result_underid.value.msg)

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
        result_small = @test_throws ErrorException estimate(DifferenceGMM(), diff_small)
        @test occursin("Insufficient observations", result_small.value.msg)

        # Collinear regressors (rank(X) < n_regressors) must be rejected with
        # an actionable message, not surface as a bare
        # LinearAlgebra.SingularException deep inside the GMM solver.
        X_collinear = hcat(X_mat, 2.0 .* X_mat)
        diff_collinear = merge(diff_data, (X=X_collinear, coef_names=["L.y", "L.y2"]))
        result_collinear = @test_throws ErrorException estimate(DifferenceGMM(), diff_collinear)
        @test occursin("collinear", result_collinear.value.msg)
    end

    # Guards the Arellano-Bond (1991, p.279) one-step weighting matrix:
    # A_N = N^-1 * sum_i Z_i'HZ_i, H_i[t,s] = 2 (t==s), -1 (calendar-adjacent).
    @testset "Arellano-Bond H weighting matrix" begin
        # Two individuals, three consecutive differenced-equation periods each
        # (times 2,3,4), one instrument column of all-ones so Z'HZ reduces to
        # sum of H_i entries.
        Z = sparse(ones(6, 1))
        panel_info_h = [
            (id=1, time=2, is_level=false),
            (id=1, time=3, is_level=false),
            (id=1, time=4, is_level=false),
            (id=2, time=2, is_level=false),
            (id=2, time=3, is_level=false),
            (id=2, time=4, is_level=false),
        ]
        # Per individual: H_i is 3x3 with 2 on the diagonal and -1 on
        # calendar-adjacent off-diagonals -> row/col sums are 2-1=1 (edges) or
        # 2-1-1=0 (middle); total over the 3x3 all-ones contraction is
        # sum(H_i) = 3*2 + 4*(-1) = 2 per individual, summed over both.
        H = DynamicPanelModels._ab_h_weight_matrix(Z, panel_info_h)
        @test H[1, 1] ≈ 4.0  # 2 individuals * (sum of one 3x3 H_i = 2)

        # Gaps in an unbalanced panel must not get an off-diagonal -1: individual
        # 2 is missing time 3, so times 2 and 4 are not calendar-adjacent.
        panel_info_gap = [
            (id=1, time=2, is_level=false),
            (id=1, time=3, is_level=false),
            (id=2, time=2, is_level=false),
            (id=2, time=4, is_level=false),
        ]
        Z_gap = sparse(ones(4, 1))
        H_gap = DynamicPanelModels._ab_h_weight_matrix(Z_gap, panel_info_gap)
        # individual 1 (adjacent 2,3): sum(H_1) = 2*2 - 2*1 = 2
        # individual 2 (gap, not adjacent): sum(H_2) = 2*2 + 0 = 4
        @test H_gap[1, 1] ≈ 6.0

        # DifferenceGMM dispatches to the H-weighted matrix; other estimators
        # keep the plain (Z'Z)^-1 fallback.
        diff_data_h = merge(diff_data, (panel_info=panel_info_h, n_groups=2))
        W_ab = DynamicPanelModels.initial_weight_matrix(DifferenceGMM(), Z, diff_data_h)
        @test W_ab ≈ inv(H / 2)
        W_plain = DynamicPanelModels.initial_weight_matrix(SystemGMM(), Z, diff_data_h)
        @test W_plain ≈ inv(Matrix(Z' * Z))
    end

    # Regression guard for the densify-once fix in calculate_clustered_weight_matrix
    # (previously row-sliced a sparse Z per cluster): checks the actual numeric
    # output against an independently-computed per-cluster sum of outer products.
    @testset "calculate_clustered_weight_matrix numeric" begin
        Z_cw = sparse(reshape([1.0, 1, 1, 1, 2, 2, 2, 2], 4, 2))
        res_cw = [0.1, -0.2, 0.3, -0.1]
        Ω = DynamicPanelModels.calculate_clustered_weight_matrix(Z_cw, res_cw, panel_info, 2)
        Zm = Matrix(Z_cw)
        c1 = Zm[1:2, :]' * res_cw[1:2]
        c2 = Zm[3:4, :]' * res_cw[3:4]
        Ω_expected = c1 * c1' + c2 * c2'
        @test Ω ≈ Ω_expected

        # Guard that the `ranges` kwarg is actually consumed, not silently
        # ignored in favor of recomputing _cluster_ranges(panel_info)
        # internally: passing a deliberately different partition (one big
        # cluster instead of two) must change the result.
        wrong_ranges = [1:4]
        Ω_wrong = DynamicPanelModels.calculate_clustered_weight_matrix(
            Z_cw, res_cw, panel_info, 2; ranges=wrong_ranges
        )
        c_all = Zm[1:4, :]' * res_cw[1:4]
        @test Ω_wrong ≈ c_all * c_all'
        @test !isapprox(Ω_wrong, Ω_expected)
    end

    # Regression guard for the G1 = ZtX' * W1 hoist in _solve_gmm: the one-step
    # coefficient must match an independent, unhoisted recomputation of the sandwich.
    @testset "One-step coefficient matches independent computation" begin
        result = estimate(DifferenceGMM(), diff_data; steps=1, robust=true)
        # estimate() drops collinear instrument columns by default; replicate
        # that step so Z matches what _solve_gmm actually saw.
        Z = DynamicPanelModels._drop_collinear_columns(
            DynamicPanelModels.build_instruments(DifferenceGMM(), diff_data)
        )
        W1 = DynamicPanelModels.initial_weight_matrix(DifferenceGMM(), Z, diff_data)
        ZtX = Z' * X_mat
        bread = inv(ZtX' * W1 * ZtX)
        β_check = bread * (ZtX' * W1 * (Z' * y))
        @test β_check ≈ result.coef atol=1e-8
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
