.PHONY: test test-cov format docs example

test:
	julia --project=. -e 'using Pkg; Pkg.test()'

test-cov:
	julia --project=. --code-coverage=user -e 'using Pkg; Pkg.test(coverage=true)'

format:
	julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.add("JuliaFormatter"); using JuliaFormatter; format("src"); format("test")'

docs:
	julia --project=docs docs/make.jl

example:
	julia --project=examples examples/example.jl
