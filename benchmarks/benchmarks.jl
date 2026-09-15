# benchmarks/benchmarks.jl
#
# Ad-hoc timing/allocation checks for the GMM solver on a larger panel, to
# catch regressions in the BLAS-fused, in-place (`.+=`/`mul!`, never `A += ...`)
# accumulation paths. Not wired into CI — run locally before/after
# touching estimation.jl:
#   julia --project=benchmarks benchmarks/benchmarks.jl

using Pkg
Pkg.develop(PackageSpec(path=dirname(@__DIR__)))
Pkg.instantiate()

using DynamicPanelModels
using DataFrames
using Random
using Chairmarks

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

for N in (500, 5_000, 50_000)
    df = simulate_dynamic_panel(; N=N, T=6)
    b = @be fit(
        DifferenceGMM(; robust=true, steps=2),
        $df;
        formula="y ~ lag(y) + x",
        id_col=:id,
        time_col=:t,
        exog=["x"],
    )
    s = minimum(b)
    println(
        "N=$(lpad(N, 6)): $(round(s.time * 1e3; digits=1)) ms, ",
        "$(round(s.bytes / 2^20; digits=1)) MiB, $(round(Int, s.allocs)) allocs (best of $(length(b.samples)))",
    )
end
