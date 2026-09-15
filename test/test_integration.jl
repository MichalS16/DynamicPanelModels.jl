# test/test_integration.jl
#
# End-to-end integration tests: simulate a dynamic panel with known true
# parameters and verify that DifferenceGMM / SystemGMM / AndersonHsiao recover
# them within a reasonable tolerance. These guard against regressions in the
# full fit() -> get_diff_data -> build_instruments -> estimate() pipeline that
# unit tests on small hand-built NamedTuples cannot catch.

using Test
using DynamicPanelModels
using DataFrames
using Random
using Statistics
using StatsBase: coefnames
using LinearAlgebra: diag, dot, rank

"""
    simulate_dynamic_panel(; N, T, rho, beta, seed)

Simulate a simple AR(1)-with-covariate dynamic panel:
    y_it = rho * y_i,t-1 + beta * x_it + eta_i + eps_it
with fixed effects `eta_i` and idiosyncratic shocks `eps_it`. `T` extra
"burn-in" periods are simulated and discarded so the panel starts closer to
its stationary distribution (mitigating initial-condition bias).
"""
function simulate_dynamic_panel(; N=300, T=6, rho=0.5, beta=0.8, seed=1)
    Random.seed!(seed)
    burn_in = 10
    rows = NamedTuple[]
    for i in 1:N
        eta = randn()
        y_prev = eta / (1 - rho) + randn()
        for t in (-burn_in + 1):T
            x = randn()
            eps = 0.5 * randn()
            y = rho * y_prev + beta * x + eta + eps
            if t >= 1
                push!(rows, (id=i, t=t, y=y, x=x))
            end
            y_prev = y
        end
    end
    return DataFrame(rows)
end

@testset "Integration: recover known parameters" begin
    df = simulate_dynamic_panel(; N=400, T=6, rho=0.5, beta=0.8, seed=1)

    @testset "DifferenceGMM one-step" begin
        m = fit(
            DifferenceGMM(; robust=true, steps=1),
            df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
        )
        b = coef(m)
        @test isapprox(b[1], 0.5; atol=0.15)
        @test isapprox(b[2], 0.8; atol=0.15)
        @test nobs(m) > 0
        @test ngroups(m) == 400
    end

    @testset "DifferenceGMM two-step differs from one-step" begin
        m1 = fit(
            DifferenceGMM(; robust=true, steps=1),
            df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
        )
        m2 = fit(
            DifferenceGMM(; robust=true, steps=2),
            df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
        )
        # Two-step re-weights with the estimated optimal weight matrix, so the
        # coefficient vector should generally differ from one-step.
        @test coef(m1) != coef(m2)
        @test m2.windmeijer == true
        @test all(diag(vcov(m2)) .> 0)
        b2 = coef(m2)
        @test isapprox(b2[1], 0.5; atol=0.2)
        @test isapprox(b2[2], 0.8; atol=0.2)
    end

    @testset "SystemGMM recovers parameters" begin
        m = fit(
            SystemGMM(; robust=true, steps=2),
            df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
        )
        b = coef(m)
        @test isapprox(b[1], 0.5; atol=0.15)
        @test isapprox(b[2], 0.8; atol=0.15)
    end

    @testset "SystemGMM level-equation intercept avoids bias with nonzero means" begin
        # A fixed effect and regressor with nonzero unconditional mean is the
        # realistic case (e.g. log employment/output levels): without a
        # level-equation intercept, the level moment E[(y_it - β'x_it)Δy_{i,t-1}]
        # = 0 is violated whenever E[η_i] != 0, biasing β̂.
        function simulate_nonzero_mean(;
            N=400, T=7, rho=0.5, beta=0.8, eta_mean=3.0, x_mean=2.0, seed=7
        )
            Random.seed!(seed)
            rows = NamedTuple[]
            for i in 1:N
                eta = eta_mean + randn()
                y_prev = (eta + beta * x_mean) / (1 - rho) + randn()
                for t in -9:T
                    x = x_mean + randn()
                    eps = 0.5 * randn()
                    y = rho * y_prev + beta * x + eta + eps
                    t >= 1 && push!(rows, (id=i, t=t, y=y, x=x))
                    y_prev = y
                end
            end
            return DataFrame(rows)
        end
        df_nz = simulate_nonzero_mean()
        m = fit(
            SystemGMM(; robust=true, steps=2),
            df_nz;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
        )
        b = coef(m)
        @test isapprox(b[1], 0.5; atol=0.15)
        @test isapprox(b[2], 0.8; atol=0.15)
    end

    @testset "vcov is exactly symmetric (not just diagonal-correct)" begin
        # wald_test and ar_test both read the *full* vcov matrix (R*V*R',
        # dot(x, V*x)), not just its diagonal. Chained generic `inv()` calls
        # in _solve_gmm don't preserve exact float symmetry, most visible
        # near weak identification (SystemGMM, rho close to 1, with
        # time_effects adding more near-collinear instruments): found via
        # randomized fuzzing across (estimator, steps, robust, time_effects)
        # combinations, reproduced deterministically here with SystemGMM
        # steps=2 robust=false at rho=0.93.
        df_weak = simulate_dynamic_panel(; N=67, T=6, rho=0.93, beta=-0.13, seed=22)
        for spec in (
            DifferenceGMM(; robust=false, steps=1),
            DifferenceGMM(; robust=true, steps=1),
            DifferenceGMM(; robust=false, steps=2),
            DifferenceGMM(; robust=true, steps=2),
            SystemGMM(; robust=false, steps=1),
            SystemGMM(; robust=false, steps=2),
            SystemGMM(; robust=true, steps=2),
        )
            m = fit(
                spec,
                df_weak;
                formula="y ~ lag(y) + x",
                id_col=:id,
                time_col=:t,
                exog=["x"],
                time_effects=true,
            )
            @test m.vcov == m.vcov'
        end
    end

    @testset "AndersonHsiao recovers parameters (noisier)" begin
        m = fit(AndersonHsiao(), df; formula="y ~ lag(y) + x", id_col=:id, time_col=:t, exog=["x"])
        b = coef(m)
        @test isapprox(b[1], 0.5; atol=0.3)
        @test isapprox(b[2], 0.8; atol=0.3)

        # One level instrument per lag(y, k): lag(y, 2) adds y_{t-3}, so the
        # model stays exactly identified (3 instruments / 3 regressors); on an
        # AR(1) DGP the second lag's coefficient is ~0. Must equal a hand 2SLS.
        m2 = fit(
            AndersonHsiao(),
            df;
            formula="y ~ lag(y) + lag(y, 2) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
        )
        @test ninstruments(m2) == 3
        @test isapprox(coef(m2)[1], 0.5; atol=0.3)
        @test isapprox(coef(m2)[2], 0.0; atol=0.3)
        Zm = Matrix(m2.Z)
        P = Zm * ((Zm' * Zm) \ (Zm' * m2.X))
        @test isapprox(coef(m2), (m2.X' * P) \ (P' * m2.y); rtol=1e-8)

        # FOD shifts the instrument timing (y_{t-k}, not y_{t-k-1}) but still identifies.
        m_fod = fit(
            AndersonHsiao(),
            df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
            transform=:fod,
        )
        @test ninstruments(m_fod) == 2
        @test isapprox(coef(m_fod)[1], 0.5; atol=0.3)
    end

    @testset "time_effects with a deeper lag: FD, FOD, and SystemGMM all stay full rank" begin
        # T=6 and max lag order 2 -> dummies for period indices 4..6 (three).
        for (spec, kw) in (
            (DifferenceGMM(; robust=true, steps=2), (;)),
            (DifferenceGMM(; robust=true, steps=2), (; transform=:fod)),
            (SystemGMM(; robust=true, steps=2), (;)),
        )
            m = fit(
                spec,
                df;
                formula="y ~ lag(y) + lag(y, 2) + x",
                id_col=:id,
                time_col=:t,
                exog=["x"],
                time_effects=true,
                kw...,
            )
            @test rank(m.X) == size(m.X, 2)
            @test count(n -> startswith(n, "t="), coefnames(m)) == 3
            @test isapprox(coef(m)[1], 0.5; atol=0.2)
        end
    end

    @testset "Diagnostics run end-to-end without error" begin
        m = fit(
            DifferenceGMM(; robust=true, steps=2),
            df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
        )
        id_vec = [row.id for row in m.metadata[:panel_info]]
        time_vec = [row.time for row in m.metadata[:panel_info]]

        sargan = sargan_test(m)
        @test sargan isa DynamicPanelTest
        @test 0.0 <= sargan.pvalue <= 1.0

        ar1 = ar_test(m, 1, id_vec, time_vec)
        ar2 = ar_test(m, 2, id_vec, time_vec)
        # Independently recompute AR(1) from Arellano & Bond (1991) eq. (8)
        # directly (per-cluster cross moments, not ar_test's private
        # helpers), rather than only bounding the p-value.
        res_lag1 = DynamicPanelModels._lag_vector(m.residuals, id_vec, time_vec, 1)
        Zm = Matrix(m.Z)
        groups1 = Dict{Int,Vector{Int}}()
        for i in eachindex(id_vec)
            push!(get!(groups1, id_vec[i], Int[]), i)
        end
        term1_1 = sum(dot(m.residuals[r], res_lag1[r])^2 for r in values(groups1))
        q1 = sum(
            (Zm[r, :]' * m.residuals[r]) .* dot(m.residuals[r], res_lag1[r]) for
            r in values(groups1)
        )
        ZtX1 = Zm' * m.X
        M1 = inv(ZtX1' * m.W * ZtX1)
        Xel1 = m.X' * res_lag1
        term2_1 = -2 * dot(Xel1, M1 * ZtX1' * m.W * q1)
        term3_1 = dot(Xel1, m.vcov * Xel1)
        expected_stat1 = dot(m.residuals, res_lag1) / sqrt(term1_1 + term2_1 + term3_1)
        @test isapprox(ar1.stat, expected_stat1)
        @test 0.0 <= ar2.pvalue <= 1.0

        # Cached accessor now works after computing via the 4-arg form
        @test ar_test(m, 1) === ar1
        @test ar_test(m, 2) === ar2

        results = diagnose(m; id=id_vec, time=time_vec)
        @test haskey(results, :sargan)
        @test haskey(results, :ar1)
        @test haskey(results, :ar2)
    end

    @testset "Input interfaces agree (string / @formula / Tables source)" begin
        spec = DifferenceGMM(; robust=true, steps=2)

        m_str = fit(spec, df; formula="y ~ lag(y) + x", id_col=:id, time_col=:t, exog=["x"])
        m_fml = fit(spec, df; formula=@formula(y ~ lag(y) + x), id_col=:id, time_col=:t, exog=["x"])
        # Any Tables.jl source (here a plain column NamedTuple), not just DataFrame
        tbl = (id=df.id, t=df.t, y=df.y, x=df.x)
        m_tbl = fit(spec, tbl; formula="y ~ lag(y) + x", id_col=:id, time_col=:t, exog=["x"])

        @test coef(m_str) ≈ coef(m_fml)
        @test coef(m_str) ≈ coef(m_tbl)
        @test coefnames(m_fml) == ["L.y", "x"]

        # Non-table input is rejected with a clear error
        @test_throws ArgumentError get_diff_data([1, 2, 3], :id, :t, "y ~ lag(y)", spec)
    end

    @testset "Forward orthogonal deviations (:fod)" begin
        m_fd = fit(
            DifferenceGMM(; robust=true, steps=2),
            df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
            transform=:fd,
        )
        m_fod = fit(
            DifferenceGMM(; robust=true, steps=2),
            df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
            transform=:fod,
        )
        # FOD recovers the same structural parameters
        @test isapprox(coef(m_fod)[1], 0.5; atol=0.2)
        @test isapprox(coef(m_fod)[2], 0.8; atol=0.2)
        # FOD instruments start one lag earlier (y_{t-1}) -> at least as many as FD
        @test ninstruments(m_fod) >= ninstruments(m_fd)
        @test all(diag(vcov(m_fod)) .> 0)

        # Invalid transform / unsupported combination
        @test_throws ArgumentError get_diff_data(
            df, :id, :t, "y ~ lag(y)", DifferenceGMM(); transform=:nope
        )
        @test_throws ArgumentError get_diff_data(
            df, :id, :t, "y ~ lag(y)", SystemGMM(); transform=:fod
        )
    end

    @testset "time_effects runs end-to-end and recovers parameters" begin
        m = fit(
            DifferenceGMM(; robust=true, steps=2),
            df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
            time_effects=true,
        )
        # Structural coefficients still recovered with time dummies included
        @test isapprox(coef(m)[1], 0.5; atol=0.2)
        @test isapprox(coef(m)[2], 0.8; atol=0.2)
        # First two names are the structural regressors; the rest are period dummies
        @test coefnames(m)[1:2] == ["L.y", "x"]
        @test all(startswith.(coefnames(m)[3:end], "t="))
        @test all(diag(vcov(m)) .> 0)
    end

    @testset "windmeijer=false is actually honored (not silently ignored)" begin
        # Regression test for a bug where model.windmeijer was never read by
        # estimate()/_solve_gmm, so two-step robust SEs were always
        # Windmeijer-corrected regardless of what the user requested.
        m_on = fit(
            DifferenceGMM(; robust=true, steps=2, windmeijer=true),
            df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
        )
        m_off = fit(
            DifferenceGMM(; robust=true, steps=2, windmeijer=false),
            df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
        )
        @test m_on.windmeijer == true
        @test m_off.windmeijer == false
        # The two variance-covariance matrices must actually differ.
        @test !isapprox(vcov(m_on), vcov(m_off))
        # windmeijer=false should equal the naive two-step sandwich (steps=2,
        # robust=false uses the same bread2 fallback internally).
        m_naive = fit(
            DifferenceGMM(; robust=false, steps=2),
            df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
        )
        @test isapprox(vcov(m_off), vcov(m_naive))
    end

    @testset "SystemGMM windmeijer=false is actually honored" begin
        # Same regression as the DifferenceGMM case above, but for SystemGMM:
        # _model_windmeijer reads the same `windmeijer` field for both model
        # specs, so verify the correction is actually skipped here too, not
        # just stored on the struct.
        m_on = fit(
            SystemGMM(; robust=true, steps=2, windmeijer=true),
            df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
        )
        m_off = fit(
            SystemGMM(; robust=true, steps=2, windmeijer=false),
            df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
        )
        @test m_on.windmeijer == true
        @test m_off.windmeijer == false
        @test !isapprox(vcov(m_on), vcov(m_off))
        m_naive = fit(
            SystemGMM(; robust=false, steps=2),
            df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
        )
        @test isapprox(vcov(m_off), vcov(m_naive))
    end

    @testset "collapse and max_lags actually change the fitted result" begin
        # Regression guard for the anti-pattern of testing a kwarg
        # only via instrument-matrix column counts: confirm collapse=true and
        # a tight max_lags actually change the coefficients/vcov reaching the
        # user through fit(), not just the intermediate instrument matrix.
        spec = DifferenceGMM(; robust=true, steps=2)
        m_full = fit(spec, df; formula="y ~ lag(y) + x", id_col=:id, time_col=:t, exog=["x"])
        m_collapsed = fit(
            spec, df; formula="y ~ lag(y) + x", id_col=:id, time_col=:t, exog=["x"], collapse=true
        )
        m_capped = fit(
            spec, df; formula="y ~ lag(y) + x", id_col=:id, time_col=:t, exog=["x"], max_lags=1
        )

        @test ninstruments(m_collapsed) < ninstruments(m_full)
        @test !isapprox(coef(m_full), coef(m_collapsed))
        @test !isapprox(vcov(m_full), vcov(m_collapsed))

        @test ninstruments(m_capped) < ninstruments(m_full)
        @test !isapprox(coef(m_full), coef(m_capped))
        @test !isapprox(vcov(m_full), vcov(m_capped))

        # Collapsed instruments still recover the true parameters reasonably
        @test isapprox(coef(m_collapsed)[1], 0.5; atol=0.2)
        @test isapprox(coef(m_collapsed)[2], 0.8; atol=0.2)
    end

    @testset "is_robust / is_windmeijer report distinct, correct flags" begin
        m_1step_robust = fit(
            DifferenceGMM(; robust=true, steps=1),
            df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
        )
        m_2step_wind = fit(
            DifferenceGMM(; robust=true, steps=2, windmeijer=true),
            df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
        )
        m_1step_naive = fit(
            DifferenceGMM(; robust=false, steps=1),
            df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
        )
        # Regression test: one-step robust SEs are robust even though the
        # Windmeijer correction (a two-step-only concept) was never applied.
        @test is_robust(m_1step_robust) == true
        @test is_windmeijer(m_1step_robust) == false
        @test is_robust(m_2step_wind) == true
        @test is_windmeijer(m_2step_wind) == true
        @test is_robust(m_1step_naive) == false
        @test is_windmeijer(m_1step_naive) == false
    end

    @testset "duplicate (id, time) rows are rejected, not silently corrupted" begin
        bad = DataFrame(id=[1, 1, 1, 1, 1], t=[1, 2, 2, 3, 4], y=[1.0, 2.0, 20.0, 3.0, 4.0])
        @test_throws ErrorException get_diff_data(bad, :id, :t, "y ~ lag(y)", DifferenceGMM())
        @test_throws ErrorException fit(
            DifferenceGMM(), bad; formula="y ~ lag(y)", id_col=:id, time_col=:t
        )
    end

    @testset "Windmeijer correction sanity: two-step robust SEs are well-scaled" begin
        # Regression guard for a prior bug where the Windmeijer correction's D
        # matrix was assembled incorrectly, inflating two-step robust SEs by
        # orders of magnitude (observed: SE ~9-23 instead of ~0.02-0.5) and, as a
        # consequence, collapsing the Arellano-Bond AR(1) test statistic toward
        # zero (non-significant) even though the residuals were i.i.d.-in-levels
        # and AR(1) should be strongly significant by construction.
        wm_df = simulate_dynamic_panel(; N=400, T=8, rho=0.5, beta=0.8, seed=7)
        m1 = fit(
            DifferenceGMM(; robust=true, steps=1),
            wm_df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
        )
        m2 = fit(
            DifferenceGMM(; robust=true, steps=2),
            wm_df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
        )
        se1, se2 = stderror(m1), stderror(m2)

        # Two-step Windmeijer SEs must stay within the same order of magnitude
        # as the one-step robust SEs, not blow up by 10x+ (the symptom of the bug).
        @test all(se2 ./ se1 .< 5.0)
        @test all(se2 .< 2.0)

        id_vec = [r.id for r in m2.metadata[:panel_info]]
        time_vec = [r.time for r in m2.metadata[:panel_info]]
        ar1 = ar_test(m2, 1, id_vec, time_vec)
        ar2 = ar_test(m2, 2, id_vec, time_vec)
        # By construction (i.i.d. eps_it), differenced residuals have a strong
        # negative first-order autocorrelation: AR(1) must be significant.
        @test ar1.pvalue < 0.01
        # No AR(2) in the DGP: should not be significant at a conservative
        # (1%) level. This is a random draw, so occasional mild significance at
        # the 5% level is expected noise, not a sign of the bug this test
        # guards against (which manifested as p -> 1 from inflated variance,
        # not spurious low p-values).
        @test ar2.pvalue > 0.01
    end

    @testset "One-step Sargan J-statistic has the right chi-squared scale" begin
        # Guards two distinct historical bugs that both left this stat far
        # from its degrees of freedom: falling back to plain Z'Z (no H matrix
        # at all) left it near zero; the H matrix present but the weight
        # matrix wrongly divided by N (see `_one_step_variance`'s docstring)
        # inflated it to N times too large instead.
        sargan_df = simulate_dynamic_panel(; N=400, T=8, rho=0.5, beta=0.8, seed=11)
        m = fit(
            DifferenceGMM(; robust=true, steps=1),
            sargan_df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
        )
        sargan = sargan_test(m)
        @test sargan.stat > 0.2 * sargan.dof

        # Same guard for the (Z'Z)^-1 one-step estimators: e'Z(Z'Z)^-1Z'e alone
        # (no σ̂² division) is ~σ̂² × too small (was ≈1.2 for 38 df on EmplUK).
        m_sys = fit(
            SystemGMM(; robust=true, steps=1),
            sargan_df;
            formula="y ~ lag(y) + x",
            id_col=:id,
            time_col=:t,
            exog=["x"],
        )
        sargan_sys = sargan_test(m_sys)
        @test sargan_sys.stat > 0.2 * sargan_sys.dof
        # and an independent formula check of the scale actually applied
        Zs = Matrix(m_sys.Z)
        es = m_sys.residuals
        qs = dot(Zs' * es, (Zs' * Zs) \ (Zs' * es))
        σ2s = dot(es, es) / (m_sys.n_obs - length(coef(m_sys)))
        @test isapprox(sargan_sys.stat, qs / σ2s; rtol=1e-6)
    end
end
