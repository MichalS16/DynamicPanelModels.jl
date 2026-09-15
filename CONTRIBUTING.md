# Contributing

## Workflow

```bash
make hooks     # one-time: enable the versioned git hooks in .githooks/
make test      # full test suite (Aqua + unit + integration + AB(1991) replication)
make format    # format src/ and test/ (blue style, margin 100) — CI rejects unformatted code
make docs      # build the Documenter docs locally
```

The hooks format staged Julia files on commit, reject wip/fixup commit
messages on `main`, and run the test suite before every push.

## Where things live

Estimation pipeline: `fit` → `get_diff_data` (`transformations.jl`) →
`build_instruments` (`instruments.jl`) → `estimate` / `_solve_gmm`
(`estimation.jl`) → diagnostics (`diagnostics.jl`). Tests mirror `src/`
one-to-one, plus `test/test_integration.jl` (simulated DGPs with known
parameters) and `test/test_replication.jl` (Arellano-Bond 1991 Table 4 on the
bundled EmplUK data, checked against the published values and R's `plm`).

## Econometric changes

Correctness against the source papers is the priority — Arellano & Bond
(1991), Blundell & Bond (1998), Windmeijer (2005), Roodman (2009). A change
that touches estimation or a test statistic needs an *independent* check
(a hand-derived value, a published number, or a reference implementation such
as `plm`/`xtabond2`), not a test that calls the same production function on
both sides of an `isapprox`. `test/test_replication.jl` must keep passing to
1e-4 — if a change legitimately moves those numbers, explain why in the PR.

Add every new public function to the `export` list in
`src/DynamicPanelModels.jl` and to the matching `@docs` block in
`docs/src/api.md`, and note user-visible changes in `CHANGELOG.md`.
