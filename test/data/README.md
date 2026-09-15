# Test data

## EmplUK.csv

The Arellano & Bond (1991) UK employment panel: 140 firms, 1976-1984
(unbalanced, 1031 firm-years), with employment, wages, capital and output.
This is the dataset behind Table 4 of the paper and the `abdata` used in
Stata's `xtabond`/`xtabond2` documentation.

Copied unchanged from the `plm` R package's `EmplUK` data set, via the
Rdatasets CSV mirror
(<https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/plm/EmplUK.csv>).
Bundled so that `test/test_replication.jl` can check the estimators against the
published Table 4 values and against `plm::pgmm` without network access.
