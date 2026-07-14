# test/runtests.jl

# libraries
using DynamicPanelModels
using Test
using Aqua
using DataFrames
using Statistics
using Random
using SparseArrays
using LinearAlgebra

# Set fix random seed
Random.seed!(123)

@testset "DynamicPanelModels.jl" begin
    # Quality Checks
    @testset "Code Quality" begin
        Aqua.test_all(DynamicPanelModels; ambiguities=true, stale_deps=true, deps_compat=true)
    end

    # Run Individual Test files
    @testset "Individual Tests" begin
        include("test_diagnostics.jl")
        include("test_estimation.jl")
        include("test_instruments.jl")
        include("test_interface.jl")
        include("test_plot_recipes.jl")
        include("test_show.jl")
        include("test_transformations.jl")
        include("test_types.jl")
        include("test_integration.jl")
    end
end
