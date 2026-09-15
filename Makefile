.PHONY: test test-cov format docs example hooks clean

hooks:
	git config core.hooksPath .githooks

# Remove per-line coverage artifacts left behind by `make test-cov`.
clean:
	find . -name '*.cov' -not -path './.git/*' -delete

test:
	julia --project=. -e 'using Pkg; Pkg.test()'

test-cov:
	julia --project=. --code-coverage=user -e 'using Pkg; Pkg.test(coverage=true)'

# Same persistent formatter environment as .githooks/pre-commit (installed once).
format:
	julia -e 'using Pkg; Pkg.activate("dpm-formatter"; shared=true, io=devnull); haskey(Pkg.project().dependencies, "JuliaFormatter") || Pkg.add("JuliaFormatter"); using JuliaFormatter; format("src"); format("test")'

# docs/ and examples/ are separate environments that need the package dev'd
# into them; Pkg.develop is idempotent, so this is safe to run every time.
docs:
	julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
	julia --project=docs docs/make.jl

example:
	julia --project=examples -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
	julia --project=examples examples/example.jl
