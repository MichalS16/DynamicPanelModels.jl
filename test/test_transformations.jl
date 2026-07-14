# test/test_transformations.jl

# libraries
using Test
using DataFrames
using DynamicPanelModels

@testset "Transformations" begin
    # Formula Parsing
    @testset "Formula Parsing" begin
        # Valid cases
        y, x = parse_formula("y ~ lag(y) + x1")
        @test y == "y"
        @test x == ["lag(y)", "x1"]

        # Another valid case
        y2, x2 = parse_formula("gdppc ~ lag(gdppc) +  investment + schooling ")
        @test y2 == "gdppc"
        @test x2 == ["lag(gdppc)", "investment", "schooling"]

        # Invalid cases
        @test_throws ErrorException parse_formula("y ~ ")
        @test_throws ErrorException parse_formula(" ~ x")
        @test_throws ErrorException parse_formula("y ~ x + y")
        @test_throws ErrorException parse_formula("y ~ x + x")
        @test_throws ErrorException parse_formula("")           # empty string
        @test_throws ErrorException parse_formula("   ")        # whitespace only
        @test_throws ErrorException parse_formula("y + x")      # no '~'
        @test_throws ErrorException parse_formula("y ~ x ~ z")  # too many '~'
        @test_throws ErrorException parse_formula("y ~ + +")    # no valid regressors

        # Empty terms between '+' are skipped, not errors
        y3, x3 = parse_formula("y ~ x1 + + x2")
        @test x3 == ["x1", "x2"]
    end

    # StatsModels @formula parsing (delegates to string parser)
    @testset "Formula Parsing (@formula)" begin
        y, x = parse_formula(@formula(y ~ lag(y) + x1))
        @test y == "y"
        @test x == ["lag(y)", "x1"]

        # Same result as the equivalent string form
        @test parse_formula(@formula(gdppc ~ lag(gdppc) + investment)) ==
            parse_formula("gdppc ~ lag(gdppc) + investment")

        # `lag` is only a formula marker; calling it directly errors
        @test_throws ArgumentError lag(1.0)
    end

    # Difference GMM Data Preparation
    @testset "Difference GMM Prep" begin
        # Simple panel data
        df = DataFrame(
            id=repeat([1, 2], inner=4),
            t=repeat([1, 2, 3, 4], 2),
            y=[1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0],
            x=[10.0, 20.0, 30.0, 40.0, 100.0, 110.0, 120.0, 130.0],
        )
        data = get_diff_data(df, :id, :t, "y ~ x", DifferenceGMM())

        # Tests on output: with T=4 per group, valid diffs exist at t=2,3,4
        @test data.n_obs == 6
        @test data.n_groups == 2
        @test all(data.y .== 1.0)
        @test all(data.X .== 10.0)
        @test all([row.is_level == false for row in data.panel_info])
        @test data.id_time_to_y[1][2] == 2.0
        @test sort([row.time for row in data.panel_info if row.id == 1]) == [2, 3, 4]
    end

    # System GMM Data Preparation
    @testset "System GMM Prep" begin
        # Simple panel data
        df = DataFrame(
            id=repeat([1], inner=4),
            t=[1, 2, 3, 4],
            y=[1.0, 2.0, 3.0, 4.0],
            x=[10.0, 20.0, 30.0, 40.0],
        )
        data = get_diff_data(df, :id, :t, "y ~ x", SystemGMM())

        # Tests on output: diffs valid at t=2,3,4; level eq. valid at t=3,4
        @test data.n_obs == 5

        # Extract rows by type
        diff_rows = [i for (i, row) in enumerate(data.panel_info) if !row.is_level]
        level_rows = [i for (i, row) in enumerate(data.panel_info) if row.is_level]

        # Check counts
        @test length(diff_rows) == 3
        @test length(level_rows) == 2
        @test all(data.y[diff_rows] .== 1.0)
        @test sort(data.y[level_rows]) == [3.0, 4.0]
        @test haskey(data.id_time_to_diff_y, 1)
        @test haskey(data.id_time_to_diff_y[1], 3)
        @test data.id_time_to_diff_y[1][3] == 1.0
    end

    # Lag Generation Logic
    @testset "Lag handling" begin
        # Simple panel with missing times
        df = DataFrame(id=[1, 1, 1, 1], t=[1, 2, 3, 4], y=[1.0, 2.0, 3.0, 4.0])
        data = get_diff_data(df, :id, :t, "y ~ lag(y)", DifferenceGMM())

        # Tests on output
        @test all(data.X .== 1.0)
    end

    # Exogenous regressor specification
    @testset "Exogenous regressors" begin
        df = DataFrame(
            id=repeat([1, 2], inner=4),
            t=repeat([1, 2, 3, 4], 2),
            y=[1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0],
            x=[10.0, 20.0, 30.0, 40.0, 100.0, 110.0, 120.0, 130.0],
        )

        # Default: no exogenous regressors declared
        data_default = get_diff_data(df, :id, :t, "y ~ lag(y) + x", DifferenceGMM())
        @test data_default.exog_idx == Int[]

        # Declaring 'x' as exogenous resolves to its column index (2nd RHS term)
        data_exog = get_diff_data(df, :id, :t, "y ~ lag(y) + x", DifferenceGMM(); exog=["x"])
        @test data_exog.exog_idx == [2]

        # Unknown exogenous term errors
        @test_throws ErrorException get_diff_data(
            df, :id, :t, "y ~ lag(y) + x", DifferenceGMM(); exog=["z"]
        )

        # The dependent variable (or its lag) cannot be marked exogenous
        @test_throws ErrorException get_diff_data(
            df, :id, :t, "y ~ lag(y) + x", DifferenceGMM(); exog=["L.y"]
        )

        # A lag of a strictly exogenous covariate MAY be exogenous
        data_lagexog = get_diff_data(
            df, :id, :t, "y ~ lag(y) + lag(x, 0:1)", DifferenceGMM(); exog=["x", "L.x"]
        )
        @test data_lagexog.exog_idx == [2, 3]
    end

    # Transformations, multiple lags, and time effects
    @testset "Transforms, multi-lag, time effects" begin
        # Positive data so log is well-defined
        df = DataFrame(
            id=repeat([1, 2], inner=5), t=repeat(1:5, 2), y=Float64.(2:11), x=Float64.(12:21)
        )

        # log() transform on both sides, nested in lag
        d1 = get_diff_data(df, :id, :t, "log(y) ~ lag(log(y)) + log(x)", DifferenceGMM())
        @test d1.coef_names == ["L.log(y)", "log(x)"]
        # differenced log-response equals Δlog(y)
        @test d1.y[1] ≈ log(df.y[3]) - log(df.y[2])

        # @formula agrees with the string form
        d1f = get_diff_data(df, :id, :t, @formula(log(y) ~ lag(log(y)) + log(x)), DifferenceGMM())
        @test d1f.coef_names == d1.coef_names
        @test d1f.X ≈ d1.X

        # Range lags expand into separate columns; explicit lag order
        d2 = get_diff_data(df, :id, :t, "y ~ lag(y) + lag(x, 0:1)", DifferenceGMM())
        @test d2.coef_names == ["L.y", "x", "L.x"]
        d3 = get_diff_data(df, :id, :t, "y ~ lag(y) + lag(x, 2)", DifferenceGMM())
        @test d3.coef_names == ["L.y", "L2.x"]

        # A lag order deeper than every group's length (n_t=5 here) makes every
        # row NaN for that column, leaving no valid observations at all -- this
        # correctly surfaces as an error rather than a silently empty/wrong result.
        @test_throws ErrorException get_diff_data(
            df, :id, :t, "y ~ lag(y) + lag(x, 10)", DifferenceGMM()
        )

        # When the lag is out-of-range for only SOME groups (id=2 has just 2
        # periods, so lag(x,3) never has a valid value there), those groups
        # contribute nothing but other groups (id=1, 5 periods) still do.
        mixed_df = DataFrame(
            id=[1, 1, 1, 1, 1, 2, 2],
            t=[1, 2, 3, 4, 5, 1, 2],
            y=[2.0, 3, 4, 5, 6, 10, 11],
            x=[12.0, 13, 14, 15, 16, 20, 21],
        )
        d_mixed = get_diff_data(mixed_df, :id, :t, "y ~ lag(y) + lag(x, 3)", DifferenceGMM())
        @test d_mixed.n_obs > 0
        @test all(row.id == 1 for row in d_mixed.panel_info)

        # Unsupported transformation is rejected (whitelist, no eval)
        @test_throws ErrorException get_diff_data(df, :id, :t, "y ~ sin(x)", DifferenceGMM())

        # Time effects add T-2 exogenous period dummies (first two periods dropped
        # for identification in the differenced equation). Here T=5 -> t=3,4,5.
        d4 = get_diff_data(df, :id, :t, "y ~ lag(y) + x", DifferenceGMM(); time_effects=true)
        @test d4.coef_names == ["L.y", "x", "t=3", "t=4", "t=5"]
        # dummy columns (3..5) are all flagged exogenous
        @test d4.exog_idx == [3, 4, 5]
    end

    # Input validation errors
    @testset "Data validation errors" begin
        good = DataFrame(id=[1, 1, 1], t=[1, 2, 3], y=[1.0, 2.0, 3.0])

        # Empty table
        @test_throws ErrorException get_diff_data(
            DataFrame(id=Int[], t=Int[], y=Float64[]), :id, :t, "y ~ lag(y)", DifferenceGMM()
        )
        # Missing id / time columns
        @test_throws ErrorException get_diff_data(good, :nope, :t, "y ~ lag(y)", DifferenceGMM())
        @test_throws ErrorException get_diff_data(good, :id, :nope, "y ~ lag(y)", DifferenceGMM())
        # Formula references a column not present
        @test_throws ErrorException get_diff_data(good, :id, :t, "y ~ missing_col", DifferenceGMM())
        # Non-numeric variable
        bad_type = DataFrame(id=[1, 1, 1], t=[1, 2, 3], y=["a", "b", "c"])
        @test_throws ErrorException get_diff_data(bad_type, :id, :t, "y ~ lag(y)", DifferenceGMM())
        # No valid observations (every group shorter than 3 periods)
        too_short = DataFrame(id=[1, 1, 2, 2], t=[1, 2, 1, 2], y=[1.0, 2.0, 3.0, 4.0])
        @test_throws ErrorException get_diff_data(too_short, :id, :t, "y ~ lag(y)", DifferenceGMM())

        # Non-Tables input rejected
        @test_throws ArgumentError get_diff_data(42, :id, :t, "y ~ lag(y)", DifferenceGMM())
    end
end
