# Contributing

See [CLAUDE.md](CLAUDE.md) for architecture, commands, and known gotchas.

Quick start:

```bash
make test      # full test suite (Aqua + unit + integration)
make format    # format src/ and test/ (blue style, margin 100) — run before every commit
make docs      # build docs locally
```

Formatting is enforced in CI — run `make format` before committing (or let the
pre-commit hook do it, see `.githooks/`, enabled via
`git config core.hooksPath .githooks`).

Econometric correctness is the priority: verify new formulas against the
source papers (Arellano-Bond 1991, Blundell-Bond 1998, Windmeijer 2005,
Roodman 2009), not just against passing tests.
