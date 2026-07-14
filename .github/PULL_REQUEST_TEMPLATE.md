## What does this change?

## Why?

(Link the issue this addresses, if any. If this changes or adds a formula,
cite the source paper — see CONTRIBUTING.md's "econometric correctness" guideline.)

## Checklist

- [ ] `julia --project=. -e 'using Pkg; Pkg.test()'` passes locally
- [ ] Added/updated tests for any new or changed behavior (not just constructor
      checks — see CLAUDE.md's gotcha on this)
- [ ] Added a `CHANGELOG.md` entry under `[Unreleased]` for user-visible changes
- [ ] Ran the formatter (`.JuliaFormatter.toml`, Blue style) before committing
- [ ] New public functions have docstrings and are `export`ed in
      `src/DynamicPanelModels.jl`
