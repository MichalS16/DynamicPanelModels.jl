# test/ test_diagnostics.jl

# libraries
using Test
using DynamicPanelModels
using LinearAlgebra
using Statistics
using SparseArrays

@testset "Diagnostics" begin
    # Create a mock DynamicPanelResult
    n_obs = 15
    n_groups = 5
    id_vec = repeat(1:5, inner=3)
    time_vec = repeat(2:4, outer=5)
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
        j_stat,
        j_pval,
        false,
        Dict{Symbol,Any}(:method => "Mock", :formula => "y ~ x"),
    )

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
        model_justid = DynamicPanelResult(
            coef,
            vcov,
            residuals,
            fitted,
            n_obs,
            n_groups,
            2,
            ["L.y", "x1"],
            X,
            y,
            spzeros(n_obs, 2),
            Matrix{Float64}(I, 2, 2),
            j_stat,
            j_pval,
            false,
            Dict{Symbol,Any}(:method => "Mock"),
        )
        sj = sargan_test(model_justid)
        @test sj.dof == 0
        @test sj.pvalue == 1.0
    end

    # Arellano-Bond AR Test
    @testset "Arellano-Bond AR Test" begin
        # Test AR(1)
        ar1 = ar_test(model, 1, id_vec, time_vec)
        @test ar1 isa DynamicPanelTest
        @test ar1.test_name == "Arellano-Bond AR(1)"
        @test !isnan(ar1.stat)
        @test 0.0 <= ar1.pvalue <= 1.0

        # Test AR(2)
        ar2 = ar_test(model, 2, id_vec, time_vec)
        @test ar2 isa DynamicPanelTest
        @test ar2.test_name == "Arellano-Bond AR(2)"

        # Mismatched dimensions
        @test_throws ErrorException ar_test(model, 1, id_vec[1:10], time_vec[1:10])
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
    end

    # Jarque-Bera Test
    @testset "Jarque-Bera Test" begin
        # Run Jarque-Bera test
        jb = jarque_bera_test(model)
        @test jb isa DynamicPanelTest
        @test jb.test_name == "Jarque-Bera"
        @test jb.dof == 2
        @test 0.0 <= jb.pvalue <= 1.0
    end

    # Heuristics Checks
    @testset "Heuristics" begin
        # Run Pseudo R²
        r2 = goodness_of_fit(model)
        @test r2 isa Float64
        @test 0.0 <= r2 <= 1.0

        # Run Instrument Proliferation Check
        msg = check_proliferation(model)
        @test msg isa String
        @test contains(msg, "exceeds groups")
    end

    # Difference-in-Hansen (C statistic) Test
    @testset "Difference-in-Hansen Test" begin
        # Local mock builder: varies only instrument count, J statistic, and n_obs.
        mk(; n_inst, j=3.0, nobs=n_obs) = DynamicPanelResult(
            coef,
            vcov,
            residuals,
            fitted,
            nobs,
            n_groups,
            n_inst,
            ["L.y", "x1"],
            X,
            y,
            spzeros(nobs, n_inst),
            Matrix{Float64}(I, n_inst, n_inst),
            j,
            0.5,
            false,
            Dict{Symbol,Any}(:method => "Mock", :formula => "y ~ x"),
        )

        # Unrestricted model: same sample (n_obs), one extra instrument, lower J
        dh = diff_hansen_test(model, mk(; n_inst=11))
        @test dh isa DynamicPanelTest
        @test dh.test_name == "Difference-in-Hansen (C)"
        @test dh.dof == 1
        @test isapprox(dh.stat, j_stat - 3.0)
        @test 0.0 <= dh.pvalue <= 1.0

        # Not nested: unrestricted has fewer/equal instruments -> error
        @test_throws ErrorException diff_hansen_test(model, mk(; n_inst=10))

        # Different sample size (e.g. DifferenceGMM vs SystemGMM) -> error
        @test_throws ErrorException diff_hansen_test(model, mk(; n_inst=11, nobs=n_obs + 5))

        # Negative naive C is clipped to a non-rejection (C=0, p=1.0), not NaN
        dh_neg = diff_hansen_test(model, mk(; n_inst=11, j=j_stat + 10.0))
        @test dh_neg.stat == 0.0
        @test dh_neg.pvalue == 1.0
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
        model_pi = DynamicPanelResult(
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
            j_stat,
            j_pval,
            false,
            Dict{Symbol,Any}(
                :method => "Mock",
                :panel_info => [(id=id_vec[i], time=time_vec[i]) for i in 1:n_obs],
            ),
        )
        results_pi = diagnose(model_pi)
        @test haskey(results_pi, :ar1)
        @test haskey(results_pi, :ar2)
    end
end
