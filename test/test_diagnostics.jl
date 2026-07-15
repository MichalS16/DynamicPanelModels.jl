# test/ test_diagnostics.jl

# libraries
using Test
using DynamicPanelModels
using StatsAPI
using LinearAlgebra
using Statistics
using Distributions: Normal, cdf
using Random
using SparseArrays

@testset "Diagnostics" begin
    # Create a mock DynamicPanelResult
    Random.seed!(42)
    n_obs = 15
    n_groups = 5
    id_vec = repeat(1:5, inner=3)
    time_vec = repeat(2:4, outer=5)
    y = rand(n_obs)
    X = rand(n_obs, 2)
    residuals = randn(n_obs)
    fitted = y .- residuals
    coef = [0.5, -0.2]
    vcov = Matrix(0.01 * I(2))
    j_stat = 5.0
    j_pval = 0.20

    # Keyword mock builder — each testset states only what it varies.
    function mock_result(;
        res=residuals,
        fitted_vals=fitted,
        nobs=n_obs,
        n_inst=10,
        j=j_stat,
        jp=j_pval,
        metadata=Dict{Symbol,Any}(:method => "Mock", :formula => "y ~ x"),
    )
        return DynamicPanelResult(
            coef,
            vcov,
            res,
            fitted_vals,
            nobs,
            n_groups,
            n_inst,
            ["L.y", "x1"],
            X,
            y,
            spzeros(nobs, n_inst),
            Matrix{Float64}(I, n_inst, n_inst),
            j,
            jp;
            metadata=metadata,
        )
    end

    model = mock_result()

    # Sargan J-Test
    @testset "Sargan Test" begin
        # Run Sargan test
        sargan = sargan_test(model)
        @test sargan isa DynamicPanelTest
        @test sargan.test_name == "Sargan J-Test"
        @test sargan.dof == 8
        @test sargan.stat == j_stat
        @test sargan.pvalue == j_pval

        # Just-identified (instruments == regressors): df <= 0 -> trivial pass
        sj = sargan_test(mock_result(; n_inst=2))
        @test sj.dof == 0
        @test sj.pvalue == 1.0
    end

    # Arellano-Bond AR Test
    @testset "Arellano-Bond AR Test" begin
        # Test AR(1) against an independently-computed z-statistic (not just a
        # tautological [0,1] bound on the p-value).
        ar1 = ar_test(model, 1, id_vec, time_vec)
        @test ar1 isa DynamicPanelTest
        @test ar1.test_name == "Arellano-Bond AR(1)"
        @test !isnan(ar1.stat)
        res_lag = DynamicPanelModels._lag_vector(residuals, id_vec, time_vec, 1)
        X_lag = reduce(
            hcat, [DynamicPanelModels._lag_vector(X[:, i], id_vec, time_vec, 1) for i in axes(X, 2)]
        )
        d = -(X' * res_lag + X_lag' * residuals)
        expected_var = sum((residuals .* res_lag) .^ 2) + dot(d, vcov * d)
        expected_stat = dot(residuals, res_lag) / sqrt(expected_var)
        @test isapprox(ar1.stat, expected_stat)
        @test isapprox(ar1.pvalue, 2.0 * (1.0 - cdf(Normal(), abs(expected_stat))))

        # Test AR(2)
        ar2 = ar_test(model, 2, id_vec, time_vec)
        @test ar2 isa DynamicPanelTest
        @test ar2.test_name == "Arellano-Bond AR(2)"

        # Mismatched dimensions
        @test_throws ErrorException ar_test(model, 1, id_vec[1:10], time_vec[1:10])

        # Degenerate case: all-zero residuals make both the moment variance and
        # the estimation-uncertainty term exactly zero, so total_variance <= 0
        # must short-circuit to a trivial (NaN stat, dof=0, NaN p-value) result
        # instead of dividing by zero / taking sqrt of a negative number.
        ar_zero = ar_test(mock_result(; res=zeros(n_obs)), 1, id_vec, time_vec)
        @test isnan(ar_zero.stat)
        @test ar_zero.dof == 0
        @test isnan(ar_zero.pvalue)
    end

    # Wald Test
    @testset "Wald Test" begin
        # Run Wald test for H0: L.y + 0.5*x1 = 0
        R = [1.0 0.0]
        r = [0.5]
        wald = wald_test(model, R, r)

        # Verify results
        @test wald isa DynamicPanelTest
        @test wald.test_name == "Wald Test"
        @test wald.dof == 1
        @test abs(wald.stat) < 1e-10
        @test wald.pvalue > 0.99

        # Dimension mismatch: R columns must equal number of coefficients
        @test_throws ErrorException wald_test(model, [1.0 0.0 0.0], [0.5])

        # Null is false (R*β ≠ r): stat and p-value should reject, matching an
        # independently-computed statistic, not just fall in [0,1] by construction.
        r_false = [0.0]
        wald_false = wald_test(model, R, r_false)
        expected_stat = ((R * coef - r_false)' * inv(R * vcov * R') * (R * coef - r_false))[1]
        @test isapprox(wald_false.stat, expected_stat)
        @test wald_false.pvalue < 0.01

        # Near-singular restriction matrix: duplicate rows make R*V*R' exactly
        # rank-deficient (cond == Inf > 1e12), which must trigger the numerical
        # regularization branch rather than `inv` throwing a SingularException.
        R_singular = [1.0 0.0; 1.0 0.0]
        r_singular = [0.5, 0.5]
        wald_singular = wald_test(model, R_singular, r_singular)
        @test wald_singular isa DynamicPanelTest
        @test isfinite(wald_singular.stat)
        @test isfinite(wald_singular.pvalue)
    end

    # Jarque-Bera Test
    @testset "Jarque-Bera Test" begin
        # Run Jarque-Bera test
        jb = jarque_bera_test(model)
        @test jb isa DynamicPanelTest
        @test jb.test_name == "Jarque-Bera"
        @test jb.dof == 2
        @test 0.0 <= jb.pvalue <= 1.0

        # Large i.i.d. Gaussian sample: normality should not be rejected.
        Random.seed!(1)
        jb_gauss = jarque_bera_test(
            mock_result(; res=randn(5000), fitted_vals=zeros(5000), nobs=5000)
        )
        @test jb_gauss.pvalue > 0.01
    end

    # Heuristics Checks
    @testset "Heuristics" begin
        # Run Pseudo R²
        gof = goodness_of_fit(model)
        @test gof isa Float64
        @test 0.0 <= gof <= 1.0

        # goodness_of_fit delegates to StatsAPI.r2 (regression guard for that fix)
        @test goodness_of_fit(model) == StatsAPI.r2(model)

        # Perfect fit: fitted == y -> r2 == 1
        @test isapprox(goodness_of_fit(mock_result(; res=zeros(n_obs), fitted_vals=y)), 1.0)

        # Run Instrument Proliferation Check
        msg = check_proliferation(model)
        @test msg isa String
        @test contains(msg, "exceeds groups")
    end

    # Difference-in-Hansen (C statistic) Test
    @testset "Difference-in-Hansen Test" begin
        # Unrestricted model: same sample (n_obs), one extra instrument, lower J
        dh = diff_hansen_test(model, mock_result(; n_inst=11, j=3.0))
        @test dh isa DynamicPanelTest
        @test dh.test_name == "Difference-in-Hansen (C)"
        @test dh.dof == 1
        @test isapprox(dh.stat, j_stat - 3.0)
        @test 0.0 <= dh.pvalue <= 1.0

        # Not nested: unrestricted has fewer/equal instruments -> error
        @test_throws ErrorException diff_hansen_test(model, mock_result(; n_inst=10))

        # Different sample size (e.g. DifferenceGMM vs SystemGMM) -> error
        @test_throws ErrorException diff_hansen_test(
            model, mock_result(; n_inst=11, nobs=n_obs + 5)
        )

        # Negative naive C is clipped to a non-rejection (C=0, p=1.0), not NaN
        dh_neg = diff_hansen_test(model, mock_result(; n_inst=11, j=j_stat + 10.0))
        @test dh_neg.stat == 0.0
        @test dh_neg.pvalue == 1.0

        # df_u - df_r == 0 (extra instruments in `unrestricted` exactly offset by
        # extra regressors, so it isn't more over-identified than `restricted`):
        # trivial pass, not a Chisq(0) error.
        unrestricted_3reg = DynamicPanelResult(
            [0.5, -0.2, 0.1],
            Matrix(0.01 * I(3)),
            residuals,
            fitted,
            n_obs,
            n_groups,
            12,
            ["L.y", "x1", "x2"],
            hcat(X, X[:, 1]),
            y,
            spzeros(n_obs, 12),
            Matrix{Float64}(I, 12, 12),
            3.0,
            0.5,
            false,
            Dict{Symbol,Any}(:method => "Mock", :formula => "y ~ x"),
        )
        dh_zero_df = diff_hansen_test(mock_result(; n_inst=11, j=3.0), unrestricted_3reg)
        @test dh_zero_df.dof == 0
        @test dh_zero_df.pvalue == 1.0
    end

    # Diagnose Wrapper
    @testset "Diagnose Wrapper" begin
        # Run full pipeline
        results = diagnose(model; id=id_vec, time=time_vec)

        # Verify all components present
        @test haskey(results, :sargan)
        @test results[:sargan] isa DynamicPanelTest
        @test haskey(results, :ar1)
        @test results[:ar1] isa DynamicPanelTest
        @test haskey(results, :ar2)
        @test haskey(results, :jarque_bera)
        @test haskey(results, :pseudo_r2)

        # Without id/time and without :panel_info in metadata, AR tests are skipped
        results_skip = diagnose(model)
        @test haskey(results_skip, :sargan)
        @test !haskey(results_skip, :ar1)

        # With :panel_info in metadata, id/time are auto-extracted -> AR tests run
        model_pi = mock_result(;
            metadata=Dict{Symbol,Any}(
                :method => "Mock",
                :panel_info => [(id=id_vec[i], time=time_vec[i]) for i in 1:n_obs],
            ),
        )
        results_pi = diagnose(model_pi)
        @test haskey(results_pi, :ar1)
        @test haskey(results_pi, :ar2)

        # Regression guard: diagnose() must not clobber ar_test's own get!-based
        # cache with a fresh Dict — a prior bug overwrote :ar_tests unconditionally.
        model_cache = mock_result()
        pre = ar_test(model_cache, 1, id_vec, time_vec)
        results_cache = diagnose(model_cache; id=id_vec, time=time_vec)
        @test results_cache[:ar1] === pre
        @test ar_test(model_cache, 1) === pre
    end
end
