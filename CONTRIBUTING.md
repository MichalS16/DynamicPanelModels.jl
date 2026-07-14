# Contributing to DynamicPanelModels.jl

Thanks for your interest in improving DynamicPanelModels.jl! This is a Julia
package for dynamic panel data GMM estimation. Contributions of all kinds —
bug reports, documentation, tests, and features — are welcome. Please also
read the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting started

```julia
# from the repository root
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Running the test suite

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The suite includes [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl) quality
checks (ambiguities, stale/undeclared dependencies, compat bounds), unit tests
mirroring `src/` one-to-one, and `test/test_integration.jl` (a simulated panel
with known parameters). Please add tests for any new behavior and keep the suite
green before opening a pull request.

## Code style

The project follows the [Blue style](https://github.com/JuliaDiff/BlueStyle)
(margin 100), configured in `.JuliaFormatter.toml`. Format before committing:

```bash
julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.add("JuliaFormatter"); using JuliaFormatter; format("src"); format("test")'
```

Markdown files (README, CHANGELOG, this file) follow the rules in
`.markdownlint.json` (line length and duplicate-heading checks are relaxed to
accommodate the CHANGELOG's repeated `### Added` headings and HTML in badges).

## Building the documentation

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

## Guidelines

- **Econometric correctness comes first.** Verify formulas against the source
  papers (Arellano-Bond 1991, Blundell-Bond 1998, Windmeijer 2005, Roodman 2009),
  not just "tests pass". Cite the reference in comments or the PR description.
- Every new public function needs a docstring and an `export` in
  `src/DynamicPanelModels.jl`.
- Keep changes surgical and focused; add a `CHANGELOG.md` entry under
  `[Unreleased]` for user-visible changes. At release time, `[Unreleased]` is
  renamed to the new version number (with today's date) and a fresh empty
  `[Unreleased]` section is added above it — see the existing `[0.3.0]` entry
  for the expected format.
- Open an issue first for larger features so the design can be discussed.

## Commit messages

Use short, imperative-mood summaries (e.g. "Fix Windmeijer correction for
two-step GMM", not "Fixed" or "Fixing"). Keep one logical change per commit;
reference the relevant paper or issue in the body when it clarifies *why*, not
just *what*.

## Reporting bugs

Use the bug report issue template, which asks for a minimal reproducible
example (a small `DataFrame` and the `fit`/`estimate` call), the expected vs.
actual result, and your Julia and package versions (`] status DynamicPanelModels`).
For new features, use the feature request template so the design can be
discussed before implementation.
