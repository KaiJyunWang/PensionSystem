# Productivity calibration

`calibrate_productivity.R` estimates the lifecycle earnings profile
\(\bar z(s)\), the discrete productivity states, and the continuous-time
transition generator \(\Lambda\) used in `paper/extension.tex`.

## Required data

The script uses two sets of Parquet files.

### Annual personal income files

Expected filename pattern:

```text
pers_income_YYY.parquet
```

Here, `YYY` is a Republic of China calendar year. For example,
`pers_income_111.parquet` is interpreted as calendar year 2022. Each file must
contain:

| Column | Type | Use |
| --- | --- | --- |
| `idn` | Character | Person identifier used to join years and demographics. |
| `wage` | Numeric | Annual wage income. Positive values are treated as employment. |

The fake files are located in:

```text
C:/Users/kevin/fia-work/fia-infras/pers_income/sample_data
```

### Personal information file

The person file must contain:

| Column | Type | Use |
| --- | --- | --- |
| `idn` | Character | Person identifier. |
| `birth_year` | Integer | Gregorian birth year used to calculate age. |
| `valid` | Logical | When present, only valid records are retained. |

The fake file is:

```text
C:/Users/kevin/fia-work/fia-infras/pers_info/sample_data/pers_info_97_113.parquet
```

If an identifier occurs more than once in the person file, one valid record is
kept. Income observations that cannot be joined to a valid birth year are
excluded.

## Configuration

Edit the settings at the beginning of `calibrate_productivity.R` before a run.
The three principal paths are deliberately at the top of the file:

```r
output_dir <- "C:/Users/kevin/pension_system/data/productivity_calibration"
income_dir <- "C:/Users/kevin/fia-work/fia-infras/pers_income/sample_data"
person_file <- "C:/Users/kevin/fia-work/fia-infras/pers_info/sample_data/pers_info_97_113.parquet"
```

Other settings control the polynomial orders, working ages, number of employed
productivity states, DuckDB memory limit, thread count, and spill directory.
The default calibration uses ages 20 through 65, five employed states, and
polynomial orders two through four.

## Method

### Analysis panel

The income year is recovered from each filename and converted to a Gregorian
year. Age is calculated as

```text
age = income year - birth year.
```

Only observations within the configured working-age range are retained. A
positive, finite wage is classified as employment. A zero, negative, or
missing wage is classified as unemployment.

### Lifecycle earnings profile

For employed observations, the dependent variable is log annual wage. Mean log
wage is removed separately in each calendar year so aggregate wage growth and
inflation do not become part of the stationary age profile. The model's
aggregate wage variable absorbs this time-varying level.

For each requested order \(p\), the script estimates two variants with
`fixest::feols`. The baseline absorbs individual fixed effects:

\[
\log y_{it}-\overline{\log y}_{t}
= \alpha_i + \sum_{j=0}^{p}\beta_j(a_{it}-40)^j+\varepsilon_{it}.
\]

The comparison variant omits \(\alpha_i\). It is labeled `no_individual_fe` in
the coefficient output; the baseline is labeled `individual_fe`. Both variants
cluster standard errors by `idn`. The individual-FE coefficients are used to
construct the productivity residuals, state values, and transition matrix.

Age is centered at 40 internally during estimation to improve numerical
conditioning, especially for the fourth-order specification. The profile is
normalized so that \(\bar z(40)=0\), and only coefficients in ordinary powers
of age are exported. Standard errors are clustered by `idn`; the complete
clustered covariance matrix is used to transform standard errors to the
reported ordinary-age coefficients. The exported function is:

\[
\bar z(a)=\sum_{j=0}^{p}\widetilde\beta_j a^j.
\]

### Productivity states

Employed wage residuals are divided at their pooled 20th, 40th, 60th, and 80th
percentiles. The mean residual in each group becomes the corresponding finite
productivity value. States are ordered as:

| State | Interpretation |
| ---: | --- |
| 0 | Unemployment, with \(z_0=-\infty\). |
| 1 | Lowest employed residual group. |
| 2 | Second employed residual group. |
| 3 | Middle employed residual group. |
| 4 | Fourth employed residual group. |
| 5 | Highest employed residual group. |

State cutoffs and values are re-estimated for each polynomial order.

### Transition generator

Productivity transitions are counted when the same identifier is observed in
two consecutive calendar years. For distinct states \(i\) and \(j\), the
annualized transition intensity is estimated as

\[
\lambda_{ij}=\frac{N_{ij}}{E_i},
\]

where \(N_{ij}\) is the number of observed transitions from \(i\) to \(j\) and
\(E_i\) is the number of consecutive-year pairs originating in state \(i\).
The diagonal is constructed as

\[
\lambda_{ii}=-\sum_{j\ne i}\lambda_{ij},
\]

so every row of \(\Lambda\) sums to zero and all off-diagonal entries are
nonnegative. The estimator treats an observed annual state change as one jump;
annual data cannot identify multiple unobserved within-year jumps.

The output is a full six-by-six generator. It does not impose the tridiagonal
restriction used by the placeholder calibration in the solver.

## Memory management

The script requires the R packages `arrow`, `dplyr`, `dbplyr`, `fixest`,
`rlang`, `DBI`, and `duckdb`. Arrow lazily scans only the required Parquet columns. The data
steps are written as dplyr pipelines, while DuckDB executes the large joins,
aggregations, and materializations outside the R heap and spills intermediate
work to disk when necessary. The employed regression sample is collected into
R because `feols` requires an in-memory data frame; only `idn`, age, demeaned
log wage, and the centered-age powers are retained. The state and transition
panels remain out-of-core.

The main memory controls are:

```r
duckdb_memory_limit <- "4GB"
duckdb_threads <- 4L
duckdb_temp_dir <- file.path(output_dir, "duckdb_spill")
```

Lower `duckdb_memory_limit` if the data-center machine has limited shared
memory. Ensure that `duckdb_temp_dir` is on a drive with enough free space for
the joined panel and state tables. Set `keep_duckdb_database <- TRUE` only when
diagnosing a run; the normal run uses an in-memory database with disk spilling.

## Running the calibration

From the repository root, run:

```powershell
Rscript code/clean_data/calibrate_productivity.R
```

The script creates the output and spill directories when needed. Existing CSV
files with the output names below are replaced.

## Output files

| File | Contents |
| --- | --- |
| `z_bar_coefficients.csv` | Ordinary-age polynomial coefficients and individual-clustered standard errors for orders 2--4, with and without individual fixed effects. |
| `productivity_states.csv` | The six state values and observation counts for each polynomial order. |
| `lambda_matrix.csv` | Long-form \(\Lambda\), transition counts, and origin exposure for every order and state pair. |
| `calibration_diagnostics.csv` | Residual RMSE, mean residual, estimation observations, and consecutive-year pairs. |
| `sample_summary.csv` | Overall joined-sample counts and age/year coverage. |
| `sample_summary_by_year.csv` | Annual observation counts, employment rates, and mean log wages. |

Every output CSV contains an `observations` column. Its interpretation follows
the unit of its row: person-years in the sample summaries, employed
person-years in the coefficient and diagnostic files, state-specific
person-years in the productivity-state file, and observed origin-destination
transition counts in the Lambda file. `lambda_matrix.csv` additionally reports
`origin_exposure_years`, which is the denominator used to estimate each row's
transition intensities.

In `lambda_matrix.csv`, select one polynomial order and reconstruct the matrix
with `from_state` as rows and `to_state` as columns. State numbering agrees with
`productivity_states.csv`.

## Data-center validation

Before interpreting unemployment rates or transitions, verify how the real
income files represent a person without wage income. The current code assumes
that the person-year remains in `pers_income_YYY.parquet` with a zero or missing
`wage`. If such people are absent from the income file, file absence cannot be
classified as unemployment without first defining a separate person-year
population at risk.

The fake data has positive simulated wages for every retained person-year.
Consequently it tests the complete computational pipeline but cannot validate
transitions into or out of unemployment. In fake-data output, state 0 therefore
has zero observations and zero estimated transition exposure.
