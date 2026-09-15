# test/test_replication.jl
#
# Replicates Arellano & Bond (1991), Table 4, columns (a1)/(a2) on the paper's
# own EmplUK data (bundled in test/data/, see the README there). Every number
# below is either the published value or the corresponding output of R's
# plm::pgmm / plm::vcovHC / plm::mtest on the same specification, so this
# anchors the whole pipeline to external references rather than to the
# package's own formulas.

using Test
using DynamicPanelModels
using DataFrames

function load_empluk()
    lines = readlines(joinpath(@__DIR__, "data", "EmplUK.csv"))
    header = String.(split(lines[1], ','))
    cols = Dict(name => Float64[] for name in header)
    for line in lines[2:end]
        for (name, val) in zip(header, split(line, ','))
            push!(cols[name], parse(Float64, val))
        end
    end
    return DataFrame(;
        firm=Int.(cols["firm"]),
        year=Int.(cols["year"]),
        n=log.(cols["emp"]),
        w=log.(cols["wage"]),
        k=log.(cols["capital"]),
        ys=log.(cols["output"]),
    )
end

@testset "Replication: Arellano-Bond (1991) Table 4 on EmplUK" begin
    df = load_empluk()
    @test nrow(df) == 1031

    # Employment equation (16): two lags of employment, current and lagged
    # wages, capital and output (capital/output with two lags), year dummies.
    f = "n ~ lag(n) + lag(n,2) + w + lag(w) + k + lag(k) + lag(k,2) + ys + lag(ys) + lag(ys,2)"
    ex = ["w", "L.w", "k", "L.k", "L2.k", "ys", "L.ys", "L2.ys"]
    fitab(spec) = fit(spec, df; formula=f, id_col=:firm, time_col=:year, exog=ex, time_effects=true)
    panel_ids(m) =
        ([r.id for r in m.metadata[:panel_info]], [r.time for r in m.metadata[:panel_info]])

    @testset "column (a1): one-step, robust SEs" begin
        m = fitab(DifferenceGMM(; robust=true, steps=1))
        @test ngroups(m) == 140
        @test nobs(m) == 611
        @test ninstruments(m) == 41
        @test isapprox(coef(m)[1], 0.6862; atol=1e-4)      # paper 0.686; plm 0.6862
        @test isapprox(stderror(m)[1], 0.1446; atol=1e-4)  # paper (0.145); plm 0.1446
        s = sargan_test(m)
        @test s.dof == 25
        @test isapprox(s.stat, 65.82; atol=0.05)           # paper 65.8
        id, t = panel_ids(m)
        @test isapprox(ar_test(m, 1, id, t).stat, -3.600; atol=2e-3)  # plm mtest(1)
        @test isapprox(ar_test(m, 2, id, t).stat, -0.516; atol=2e-3)  # plm mtest(2)
    end

    @testset "column (a2): two-step, Windmeijer SEs" begin
        m = fitab(DifferenceGMM(; robust=true, steps=2))
        @test is_windmeijer(m)
        b = coef(m)
        @test isapprox(b[1], 0.6287; atol=1e-4)   # n(-1)  paper 0.629
        @test isapprox(b[2], -0.0652; atol=1e-4)  # n(-2)  paper -0.065
        @test isapprox(b[3], -0.5258; atol=1e-4)  # w      paper -0.526
        @test isapprox(b[5], 0.2784; atol=1e-4)   # k      paper 0.278
        @test isapprox(b[8], 0.5919; atol=1e-4)   # ys     paper 0.592
        @test isapprox(stderror(m)[1], 0.1934; atol=1e-4)  # plm vcovHC (Windmeijer)
        s = sargan_test(m)
        @test s.dof == 25
        @test isapprox(s.stat, 31.38; atol=0.02)           # paper 31.4
        id, t = panel_ids(m)
        @test isapprox(ar_test(m, 1, id, t).stat, -2.125; atol=2e-3)
        @test isapprox(ar_test(m, 2, id, t).stat, -0.352; atol=2e-3)
    end

    @testset "Anderson-Hsiao is exactly identified with two lags of n" begin
        m = fit(AndersonHsiao(), df; formula=f, id_col=:firm, time_col=:year, exog=ex)
        @test ninstruments(m) == length(coef(m))  # y_{t-2}, y_{t-3} + 8 exogenous
    end
end
