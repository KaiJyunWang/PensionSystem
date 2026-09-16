# Taiwan Labor Insurance monthly data

`labor_insurance.csv` is a monthly panel assembled by
`scrape_labor_insurance_fund.py`. It begins in January 2009 and ends at the
latest month for which all requested source series are available.

## Construction and sources

The script combines three official Taiwan government sources:

1. The Bureau of Labor Funds (BLF) monthly Labor Insurance Fund reports supply
   the fund balance and investment return. The exact month-end balance is read
   from the allocation table rather than from its rounded narrative summary.
2. The Bureau of Labor Insurance (BLI) monthly statistical CSV files supply
   old-age-annuity and old-age lump-sum recipient counts. The annuity
   measure is the current number of beneficiaries paid in the month. The
   new-recipient measure is the number marked `首發` (first payment). The
   lump-sum measure includes only `一次請領老年給付` cases; it excludes
   `老年差額金` and `老年一次金`.
3. Ministry of the Interior (MOI) monthly population workbooks supply Taiwan's
   month-end registered population and monthly deaths. MOI annual five-year-age
   tables supply deaths and year-end population at ages 60 and older for the
   historical recipient reconstruction.

Because the agencies publish on different schedules, the final observation is
the latest month for which the next month's BLI first-payment report is also
available. Consequently, the aligned panel ends one month before the latest
common source month.

The annualized return is constructed from the monthly report's published
return. If a legacy report explicitly labels the value as annualized, it is
retained. Otherwise a year-to-date return `r` observed in month `m` is
annualized by compounding:

```text
annualized return = (1 + r)^(12 / m) - 1
```

Returns in the CSV are percentage points rather than decimal fractions. BLF did
not publish a return in the January--July 2009 monthly PDFs, so both return
columns are empty for those seven months; balances and recipient counts remain
available.

`new_monthly_pension_recipients` should be used to measure entry into the
monthly scheme. BLI records `首發` in the month in which the first pension is
approved/paid (`核付`), while an old-age pension for entitlement month `t` is
normally paid by the end of month `t+1`. The script therefore assigns the
`首發` observation published for payment month `t+1` to panel month `t`:

```text
aligned new recipients(t) = payment-month new recipients(t+1)
```

The same one-month back-shift is applied to the official series and both
estimated series. The unshifted payment-month fields are retained in the CSV
for auditability, and `new_monthly_pension_recipients_source_period` identifies
the BLI report month used. Fund balance and all other stock variables remain in
their stated calendar month. Thus a row dated `t` aligns the month-end fund
level in `t` with the cohort whose pension entitlement/application month is
`t`, as observed in the BLI report for `t+1`.

The month-to-month change in `monthly_pension_recipients` is a
net stock change: new recipients increase it, while deaths and other exits
decrease it. It therefore is not a valid substitute for the reported `首發`
count. The monthly BLI files provide payment-month `首發` from May 2022 onward,
which yields an aligned series from April 2022 onward. Before then, the script
reconstructs entries under the assumption that death is the only exit:

```text
payment-month estimated new recipients(q)
  = recipients(q) - recipients(q-1)
  + recipients(q-1) * Taiwan age-60+ monthly mortality rate(year)
  + shift
```

This payment-month estimate for `q` is then assigned to aligned panel month
`q-1`, in the same way as the reported `首發` series.

The annual 60+ death probability is calculated as Taiwan deaths at ages 60 and
older divided by the year-end population at ages 60 and older. It is converted
to a constant monthly probability as
`1 - (1 - annual_probability)^(1/12)`. The calculation uses the MOI's five-year
age groups from `60–64` through `100+`. Negative new-recipient estimates are
floored at zero.

The shift is a single additive number of recipients per month, fitted using all
months where BLI reports `首發`. Let the unshifted estimate in overlap month `t`
be `u(t)` and the reported count be `a(t)`. The script uses the least-squares
constant:

```text
shift = mean(a(t) - u(t)) over all reported overlap months
```

Thus the calibrated estimate and the reported series have the same mean over
the available validation period before integer rounding. The shift is normally
negative when the mortality/stock-change calculation systematically overstates
new recipients. It is recalculated each time the scraper runs as new official
months become available and is stored explicitly in
`new_monthly_pension_recipients_shift`.

For the currently generated dataset, the payment-month calibration overlap
contains 51 months from May 2022 through July 2026 and the fitted shift is
**-2,452.130046 recipients per month**. After rounding, the shifted estimate's
mean error over this sample is 0.00 recipients per month (mean absolute error:
176.16), compared with an unshifted mean error of 2,452.12. These figures will
change when later reported
months are added and the scraper re-estimates the shift.

This age-specific rate is closer to the relevant population than Taiwan's
overall crude mortality rate, but remains an approximation: the age and sex
composition of Labor Insurance pension recipients need not equal that of all
Taiwan residents aged 60 and older. The reported and reconstructed observations
can be distinguished with `new_monthly_pension_recipients_method`.

The calibrated reconstruction is also calculated for months with a reported `首發` value
and stored in `estimated_new_monthly_pension_recipients`. This provides a direct
validation sample from aligned month April 2022 onward. When the current year's age-specific
table is not yet available, the latest annual mortality rate is carried forward;
its source year is recorded in `taiwan_age_60plus_mortality_reference_year`.

## Codebook

| Column | Definition |
| --- | --- |
| `date` | Observation month in `YYYY-MM` format. |
| `fund` | Fund name; always `勞工保險基金`. |
| `fund_scope` | Insurance programs included in the fund. Occupational-accident insurance was separated beginning May 2022. |
| `fund_level_ntd` | Exact month-end Labor Insurance Fund balance, in NT dollars. |
| `fund_level_100m_ntd` | Fund balance in units of NT$100 million. |
| `reported_fund_level_ntd` | Rounded fund balance stated in the report narrative, converted to NT dollars. |
| `fund_ytd_return_pct` | Return printed in the monthly BLF report, in percentage points. In legacy reports that print only an annualized return, this is that published value. |
| `fund_annualized_return_pct` | Published return converted to an annualized percentage when it was not already annualized. |
| `taiwan_registered_population` | Month-end registered population of Taiwan. |
| `taiwan_deaths` | Registered deaths in Taiwan during the month. |
| `taiwan_age_60plus_annual_deaths` | Annual Taiwan deaths at ages 60+ used to calculate mortality. |
| `taiwan_age_60plus_population` | Year-end Taiwan population at ages 60+ used to calculate mortality. |
| `taiwan_age_60plus_monthly_mortality_rate` | Monthly death probability derived from the annual 60+ death/population ratio. |
| `taiwan_age_60plus_mortality_reference_year` | MOI annual-table year used for the 60+ mortality rate; the latest available year is carried forward when necessary. |
| `labor_insurance_old_age_pension_recipients` | Current recipients of Labor Insurance old-age annuity benefits; retained as a descriptive alias. |
| `monthly_pension_recipients` | Current recipients of Labor Insurance old-age annuity benefits. |
| `new_monthly_pension_recipients` | Reported `首發` recipients when available; otherwise the shifted mortality-adjusted reconstruction. It is shifted back one month from the BLI payment/report month to the entitlement/application month. |
| `estimated_new_monthly_pension_recipients` | Shifted mortality-adjusted estimate for every month, including reported months, and shifted back one month for validation against the aligned reported series. |
| `unshifted_estimated_new_monthly_pension_recipients` | Stock-change-plus-mortality estimate before applying the calibrated additive shift, shifted back one month. |
| `new_monthly_pension_recipients_shift` | Additive monthly shift fitted as the mean reported-minus-unshifted residual over all available `首發` months. |
| `reported_new_monthly_pension_recipients` | BLI `首發` value aligned to the preceding entitlement/application month; empty before April 2022. |
| `new_monthly_pension_recipients_method` | Either `reported_first_payment` or `estimated_from_stock_change_mortality_and_shift`. |
| `new_monthly_pension_recipients_source_period` | BLI payment/report month from which the aligned new-recipient fields were taken; always one month after `date`. |
| `new_monthly_pension_recipients_source_url` | BLI annuity-statistics CSV URL for that source period. |
| `payment_month_new_monthly_pension_recipients` | Preferred new-recipient value before temporal alignment, in the row's payment/report month. |
| `payment_month_estimated_new_monthly_pension_recipients` | Calibrated estimate before temporal alignment, in the row's payment/report month. |
| `payment_month_unshifted_estimated_new_monthly_pension_recipients` | Uncalibrated estimate before temporal alignment, in the row's payment/report month. |
| `payment_month_reported_new_monthly_pension_recipients` | Original BLI `首發` count in the row's payment/report month. |
| `payment_month_new_monthly_pension_recipients_method` | Method associated with the payment-month preferred value. |
| `lumpsum_recipients` | `一次請領老年給付` cases paid during the month; excludes `老年差額金` and `老年一次金`. |
| `source_title` | BLF monthly report title. |
| `source_url` | BLF monthly report URL. |
| `population_source_url` | MOI population download page. |
| `pension_recipients_source_url` | BLI annuity-statistics CSV URL. |
| `lumpsum_recipients_source_url` | BLI one-time-benefit CSV URL. |

Recipient values are counts, fund values are nominal NT dollars, and returns are
percentages. The CSV is UTF-8 encoded with a byte-order mark for compatibility
with spreadsheet software.

## Running the scraper

Run commands from the repository root. Python 3 and Poppler's `pdftotext` are
required. Install the Python packages in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install pandas lxml xlrd
```

On Debian or Ubuntu, install `pdftotext` with:

```bash
sudo apt install poppler-utils
```

Then generate the data:

```bash
python3 data/scrape_labor_insurance_fund.py
```

The default output is `data/labor_insurance.csv`. Downloads are cached under
`data/labor_insurance_cache` while the scraper runs and that directory is
removed after a successful CSV write. The cache remains in place if the process
fails, which makes diagnosis and a retry faster.

Useful options include:

```bash
python3 data/scrape_labor_insurance_fund.py \
  --start 2009-01 \
  --output data/labor_insurance.csv \
  --cache data/labor_insurance_cache \
  --workers 8
```

Pass `--keep-cache` to retain downloaded source files after success:

```bash
python3 data/scrape_labor_insurance_fund.py --keep-cache
```

List every command-line option with:

```bash
python3 data/scrape_labor_insurance_fund.py --help
```
