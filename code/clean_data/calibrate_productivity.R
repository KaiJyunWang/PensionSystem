# Calibrate the lifecycle earnings profile z_bar(age) and productivity generator Lambda.
#
# Required R packages: arrow, dplyr, dbplyr, fixest, rlang, DBI, duckdb
# Run from any directory with: Rscript code/clean_data/calibrate_productivity.R

# ---- Paths to edit -----------------------------------------------------------
output_dir <- "C:/Users/kevin/pension_system/data/productivity_calibration"
income_dir <- "C:/Users/kevin/fia-work/fia-infras/pers_income/sample_data"
person_file <- "C:/Users/kevin/fia-work/fia-infras/pers_info/sample_data/pers_info_97_113.parquet"

# ---- Other user settings -----------------------------------------------------
polynomial_orders <- 2:4
number_employed_states <- 5L       # Plus state 0: unemployment.
minimum_age <- 20L
maximum_age <- 65L
age_center <- 40                   # Improves conditioning of fourth-order fits.
duckdb_memory_limit <- "4GB"       # Lower this if the data-center machine is tight on RAM.
duckdb_threads <- 4L
duckdb_temp_dir <- file.path(output_dir, "duckdb_spill")
keep_duckdb_database <- FALSE      # TRUE is useful when diagnosing a long real-data run.

# Missing and nonpositive wage are treated as unemployment. The wage level is
# removed separately by calendar year, since the stationary model's wage w
# absorbs aggregate wage levels. z_bar is identified by z_bar(age_center) = 0.

required_packages <- c("arrow", "dplyr", "dbplyr", "fixest", "rlang", "DBI", "duckdb")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop("Install required package(s): ", paste(missing_packages, collapse = ", "))
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(duckdb_temp_dir, recursive = TRUE, showWarnings = FALSE)

sql_quote <- function(x) gsub("'", "''", normalizePath(x, winslash = "/", mustWork = FALSE), fixed = TRUE)
spill_path <- sql_quote(duckdb_temp_dir)

income_files <- Sys.glob(file.path(income_dir, "pers_income_*.parquet"))
if (!length(income_files)) stop("No pers_income_*.parquet files found in: ", income_dir)
if (!file.exists(person_file)) stop("Person file does not exist: ", person_file)

database_file <- if (keep_duckdb_database) {
  file.path(output_dir, "productivity_calibration.duckdb")
} else {
  ":memory:"
}

con <- DBI::dbConnect(duckdb::duckdb(), dbdir = database_file)
on.exit({
  try(DBI::dbDisconnect(con), silent = TRUE)
  if (!keep_duckdb_database && database_file != ":memory:") {
    unlink(database_file, force = TRUE)
    unlink(paste0(database_file, ".wal"), force = TRUE)
  }
}, add = TRUE)

DBI::dbExecute(con, sprintf("SET memory_limit = '%s'", duckdb_memory_limit))
DBI::dbExecute(con, sprintf("SET threads = %d", duckdb_threads))
DBI::dbExecute(con, sprintf("SET temp_directory = '%s'", spill_path))
DBI::dbExecute(con, "SET preserve_insertion_order = false")

message("Opening lazy Arrow datasets and building the disk-backed panel ...")

# Each Arrow dataset stays lazy. to_duckdb() registers its scan with DuckDB; it
# does not collect the Parquet file into the R heap.
income_arrow <- lapply(income_files, function(path) {
  roc_year <- as.integer(sub(".*pers_income_([0-9]+)\\.parquet$", "\\1", path))
  arrow::open_dataset(path) |>
    dplyr::select(idn, wage) |>
    dplyr::mutate(year = roc_year + 1911L)
})
income_tables <- lapply(income_arrow, arrow::to_duckdb,
                        con = con, auto_disconnect = FALSE)
income_source <- Reduce(dplyr::union_all, income_tables)

person_arrow <- arrow::open_dataset(person_file) |>
  dplyr::filter(dplyr::coalesce(valid, TRUE), !is.na(birth_year)) |>
  dplyr::select(idn, birth_year)
person_source <- arrow::to_duckdb(person_arrow, con = con, auto_disconnect = FALSE) |>
  dplyr::distinct(idn, .keep_all = TRUE)

analysis_panel <- income_source |>
  dplyr::inner_join(person_source, by = "idn") |>
  dplyr::mutate(
    age = year - birth_year,
    employed = dplyr::if_else(!is.na(wage) & wage > 0, 1L, 0L),
    log_wage = dplyr::if_else(!is.na(wage) & wage > 0, log(wage), NA_real_)
  ) |>
  dplyr::filter(dplyr::between(age, minimum_age, maximum_age)) |>
  dplyr::select(idn, year, age, log_wage, employed) |>
  dplyr::compute(name = "analysis_panel", temporary = FALSE)
DBI::dbExecute(con, "CREATE INDEX panel_id_year ON analysis_panel(idn, year)")

year_wage_means <- analysis_panel |>
  dplyr::filter(employed == 1L) |>
  dplyr::group_by(year) |>
  dplyr::summarise(mean_log_wage = mean(log_wage), .groups = "drop")

estimation_panel <- analysis_panel |>
  dplyr::left_join(year_wage_means, by = "year") |>
  dplyr::mutate(log_wage_demeaned = log_wage - mean_log_wage) |>
  dplyr::compute(name = "estimation_panel", temporary = TRUE)

panel_summary <- analysis_panel |>
  dplyr::summarise(
    person_years = dplyr::n(),
    people = dplyr::n_distinct(idn),
    employed_person_years = sum(employed, na.rm = TRUE),
    first_year = min(year, na.rm = TRUE), last_year = max(year, na.rm = TRUE),
    youngest_age = min(age, na.rm = TRUE), oldest_age = max(age, na.rm = TRUE)
  ) |>
  dplyr::collect()
if (!panel_summary$person_years || !panel_summary$employed_person_years) {
  stop("No usable person-year or positive-wage observations after the join.")
}
panel_summary$observations <- panel_summary$person_years
write.csv(panel_summary, file.path(output_dir, "sample_summary.csv"), row.names = FALSE)
year_summary <- analysis_panel |>
  dplyr::group_by(year) |>
  dplyr::summarise(
    person_years = dplyr::n(),
    employed_person_years = sum(employed, na.rm = TRUE),
    employment_rate = mean(employed, na.rm = TRUE),
    mean_log_wage = mean(log_wage, na.rm = TRUE),
    .groups = "drop"
  ) |>
  dplyr::arrange(year) |>
  dplyr::collect()
year_summary$observations <- year_summary$person_years
write.csv(year_summary, file.path(output_dir, "sample_summary_by_year.csv"), row.names = FALSE)

# Linear map taking coefficients in powers of (age - center) to coefficients
# in powers of age. It is also used to transform the clustered covariance.
centered_to_raw_matrix <- function(degree, center) {
  transform <- matrix(0, nrow = degree + 1L, ncol = degree + 1L)
  for (j in 0:degree) {
    for (k in 0:j) {
      transform[k + 1L, j + 1L] <- choose(j, k) * (-center)^(j - k)
    }
  }
  transform
}

coefficient_rows <- list()
state_rows <- list()
lambda_rows <- list()
diagnostic_rows <- list()

# feols requires an in-memory data frame. Collect only the columns needed by
# the regressions; all joins and the much larger state calculations stay lazy.
regression_data <- estimation_panel |>
  dplyr::filter(employed == 1L) |>
  dplyr::select(idn, age, log_wage_demeaned) |>
  dplyr::collect()
for (power in seq_len(max(polynomial_orders))) {
  regression_data[[paste0("age_centered_", power)]] <-
    (regression_data$age - age_center)^power
}

for (degree in polynomial_orders) {
  message("Calibrating polynomial order ", degree, " ...")
  regressor_names <- paste0("age_centered_", seq_len(degree))
  basis_transform <- centered_to_raw_matrix(degree, age_center)
  beta <- NULL

  for (use_individual_fe in c(TRUE, FALSE)) {
    specification <- if (use_individual_fe) "individual_fe" else "no_individual_fe"
    formula_text <- paste(
      "log_wage_demeaned ~", paste(regressor_names, collapse = " + "),
      if (use_individual_fe) "| idn" else ""
    )
    z_bar_model <- fixest::feols(
      stats::as.formula(formula_text),
      data = regression_data,
      vcov = ~idn,
      notes = FALSE,
      mem.clean = TRUE
    )

    if (use_individual_fe) {
      specification_beta <- c(
        0,
        unname(stats::coef(z_bar_model)[regressor_names])
      )
      specification_vcov <- matrix(0, degree + 1L, degree + 1L)
      specification_vcov[-1, -1] <- stats::vcov(z_bar_model)[
        regressor_names, regressor_names, drop = FALSE
      ]
    } else {
      coefficient_names <- c("(Intercept)", regressor_names)
      specification_beta <- unname(stats::coef(z_bar_model)[coefficient_names])
      specification_vcov <- stats::vcov(z_bar_model)[
        coefficient_names, coefficient_names, drop = FALSE
      ]
      # z_bar(age_center) = 0 is an exact normalization. The aggregate wage
      # absorbs the pooled intercept and its sampling variance.
      specification_beta[1] <- 0
      specification_vcov[1, ] <- 0
      specification_vcov[, 1] <- 0
    }

    raw_beta <- as.numeric(basis_transform %*% specification_beta)
    raw_vcov <- basis_transform %*% specification_vcov %*% t(basis_transform)
    coefficient_rows[[length(coefficient_rows) + 1L]] <- data.frame(
      specification = specification,
      order = degree,
      power = 0:degree,
      coefficient = raw_beta,
      std_error = sqrt(pmax(diag(raw_vcov), 0)),
      observations = stats::nobs(z_bar_model)
    )

    # The individual-FE profile is the baseline used to construct productivity
    # residuals, state values, and transition intensities below.
    if (use_individual_fe) beta <- specification_beta
    rm(z_bar_model)
    invisible(gc())
  }

  polynomial_expression <- 0
  for (power in 0:degree) {
    polynomial_expression <- rlang::expr(
      !!polynomial_expression + !!beta[power + 1L] *
        (age - !!age_center)^!!power
    )
  }
  state_table <- paste0("states_order_", degree)
  state_data <- estimation_panel |>
    dplyr::transmute(
      idn, year, age, employed,
      residual = dplyr::if_else(
        employed == 1L,
        log_wage_demeaned - !!polynomial_expression,
        NA_real_
      )
    ) |>
    dplyr::compute(name = state_table, temporary = FALSE)

  probability_grid <- (1:(number_employed_states - 1L)) / number_employed_states
  quantile_expressions <- setNames(
    lapply(probability_grid, function(probability) {
      rlang::expr(quantile(residual, !!probability))
    }),
    paste0("q", seq_along(probability_grid))
  )
  cuts <- state_data |>
    dplyr::filter(employed == 1L) |>
    dplyr::summarise(!!!quantile_expressions) |>
    dplyr::collect() |>
    unlist(use.names = FALSE)

  employed_state_expression <- number_employed_states
  for (state_number in rev(seq_along(cuts))) {
    employed_state_expression <- rlang::expr(
      dplyr::if_else(
        residual <= !!cuts[state_number],
        !!as.integer(state_number),
        !!employed_state_expression
      )
    )
  }
  states_with_class <- state_data |>
    dplyr::mutate(
      state = dplyr::if_else(employed == 0L, 0L, !!employed_state_expression)
    ) |>
    dplyr::compute(name = paste0(state_table, "_classified"), temporary = FALSE)
  DBI::dbExecute(con, sprintf(
    "CREATE INDEX %s_classified_id_year ON %s_classified(idn, year)",
    state_table, state_table
  ))

  states <- states_with_class |>
    dplyr::group_by(state) |>
    dplyr::summarise(
      observations = dplyr::n(),
      z_value = mean(residual, na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::arrange(state) |>
    dplyr::collect()
  states <- merge(
    data.frame(state = 0:number_employed_states), states,
    by = "state", all.x = TRUE, sort = TRUE
  )
  states$observations[is.na(states$observations)] <- 0
  states$order <- degree
  states$label <- ifelse(states$state == 0, "unemployment", paste0("employed_", states$state))
  states$z_value[states$state == 0] <- -Inf
  state_rows[[length(state_rows) + 1L]] <- states[, c("order", "state", "label", "z_value", "observations")]

  next_states <- states_with_class |>
    dplyr::transmute(idn, year = year - 1L, to_state = state)
  transitions <- states_with_class |>
    dplyr::select(idn, year, from_state = state) |>
    dplyr::inner_join(next_states, by = c("idn", "year")) |>
    dplyr::count(from_state, to_state, name = "transitions") |>
    dplyr::collect()
  counts <- matrix(0, number_employed_states + 1L, number_employed_states + 1L,
                   dimnames = list(0:number_employed_states, 0:number_employed_states))
  counts[cbind(transitions$from_state + 1L, transitions$to_state + 1L)] <- transitions$transitions

  # Annual observations do not reveal multiple within-year jumps. The standard
  # exposure estimator treats each observed state change as one CTMC jump:
  # q_ij = N_ij / exposure_i (i != j), q_ii = -sum_{j != i} q_ij.
  exposure <- rowSums(counts)
  lambda <- counts
  diag(lambda) <- 0
  lambda <- sweep(lambda, 1L, exposure, "/")
  lambda[!is.finite(lambda)] <- 0
  diag(lambda) <- -rowSums(lambda)
  lambda_rows[[length(lambda_rows) + 1L]] <- data.frame(
    order = degree,
    from_state = rep(0:number_employed_states, each = number_employed_states + 1L),
    to_state = rep(0:number_employed_states, times = number_employed_states + 1L),
    lambda = as.vector(t(lambda)),
    transition_count = as.vector(t(counts)),
    observations = as.vector(t(counts)),
    origin_exposure_years = rep(exposure, each = number_employed_states + 1L)
  )

  fit <- state_data |>
    dplyr::filter(employed == 1L) |>
    dplyr::summarise(
      n = dplyr::n(),
      residual_rmse = sqrt(mean(residual^2, na.rm = TRUE)),
      mean_residual = mean(residual, na.rm = TRUE)
    ) |>
    dplyr::collect()
  diagnostic_rows[[length(diagnostic_rows) + 1L]] <- data.frame(
    order = degree, n = fit$n, observations = fit$n, residual_rmse = fit$residual_rmse,
    mean_residual = fit$mean_residual, consecutive_pairs = sum(counts)
  )
  DBI::dbExecute(con, paste("DROP TABLE", paste0(state_table, "_classified")))
  DBI::dbExecute(con, paste("DROP TABLE", state_table))
  invisible(gc())
}

write.csv(do.call(rbind, coefficient_rows),
          file.path(output_dir, "z_bar_coefficients.csv"), row.names = FALSE)
write.csv(do.call(rbind, state_rows),
          file.path(output_dir, "productivity_states.csv"), row.names = FALSE)
write.csv(do.call(rbind, lambda_rows),
          file.path(output_dir, "lambda_matrix.csv"), row.names = FALSE)
write.csv(do.call(rbind, diagnostic_rows),
          file.path(output_dir, "calibration_diagnostics.csv"), row.names = FALSE)

message("Finished. CSV files written to: ", normalizePath(output_dir, winslash = "/"))
