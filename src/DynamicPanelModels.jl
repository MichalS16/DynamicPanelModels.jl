# src/DynamicPanelModels.jl

"""
    DynamicPanelModels

A Julia package for **linear dynamic panel data models** estimated via GMM:
Difference GMM (Arellano-Bond, 1991), System GMM (Blundell-Bond, 1998), and
Anderson-Hsiao (1981) IV.

Scope: this package covers *linear* dynamic panel models (a continuous
dependent variable with a lagged-dependent-variable regressor). Nonlinear
dynamic panel models (binary choice or count data with a lagged dependent
variable) are a distinct econometric problem and are out of scope.

See [`fit`](@ref), [`DifferenceGMM`](@ref), [`SystemGMM`](@ref), and
[`AndersonHsiao`](@ref) to get started.
"""
module DynamicPanelModels

# Dependencies
using DataFrames
using LinearAlgebra
using SparseArrays
using StatsBase
using Statistics
using Distributions
using StatsAPI: StatsAPI, RegressionModel
using StatsModels:
    StatsModels, @formula, FormulaTerm, Term, FunctionTerm, ConstantTerm, InterceptTerm
using Tables
using Printf
using RecipesBase

# Include source files (pipeline order: types → data prep → instruments →
# estimation → diagnostics → interface → display)
include("types.jl")
include("transformations.jl")
include("instruments.jl")
include("estimation.jl")
include("diagnostics.jl")
include("interface.jl")
include("show.jl")
include("plot_recipes.jl")

## Exports
# StatsAPI interface
export coef, vcov, stderror, residuals, fitted, predict
export nobs, dof, dof_residual, confint, coeftable, formula

# Diagnostics and Tests
export sargan_test, ar_test, wald_test, goodness_of_fit
export check_proliferation, jarque_bera_test, diagnose, diff_hansen_test

# Model types
export DifferenceGMM, SystemGMM, AndersonHsiao
export AbstractDynamicPanelModel, DynamicPanelResult, DynamicPanelTest

# Estimation and data handling
export fit, estimate
export ngroups, ninstruments, is_robust, is_windmeijer
export get_diff_data, parse_formula, build_instruments

# Formula DSL (re-exported from StatsModels) + the `lag` term for dynamic panels
export @formula, lag
end
