#!/usr/bin/env Rscript

# Reconstruct the SBGR 15-minute analysis grid from the source R data files.
# Usage: Rscript prepare_data.R <data_dir> <output_csv>
# <data_dir> is the repository data/ folder (METAR/, SIROS/, TATIC/).

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Expected <data_dir> <output_csv>")
root <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
out <- args[[2]]

load_object <- function(path, object_name) {
  env <- new.env(parent = emptyenv())
  load(file.path(root, path), envir = env)
  as.data.table(get(object_name, envir = env))
}

floor_15 <- function(x) {
  as.POSIXct(floor(as.numeric(as.POSIXct(x, tz = "UTC")) / 900) * 900,
             origin = "1970-01-01", tz = "UTC")
}

start_time <- as.POSIXct("2022-01-01 03:00:00", tz = "UTC")
end_time <- as.POSIXct("2024-06-30 23:45:00", tz = "UTC")

aggregate_actual <- function(path, time_col, prefix) {
  d <- load_object(path, "dataTF")
  d[, time := floor_15(get(time_col))]
  d <- d[time >= start_time & time <= end_time]
  d[, direction := fifelse(grepl("^(09|10)", runway), "10",
                    fifelse(grepl("^(27|28)", runway), "28", NA_character_))]
  d <- d[!is.na(direction)]
  ans <- d[, .(operations = .N,
               n10 = sum(direction == "10"),
               n28 = sum(direction == "28")), by = time]
  setnames(ans, c("operations", "n10", "n28"),
           paste0(prefix, c("_operations", "_n10", "_n28")))
  ans
}

aggregate_schedule <- function(path, time_col, name) {
  d <- load_object(path, "dataSR")
  d[, time := floor_15(get(time_col))]
  d <- d[time >= start_time & time <= end_time]
  ans <- d[, .N, by = time]
  setnames(ans, "N", name)
  ans
}

arr <- aggregate_actual("TATIC/Arrivals/dataTF_arr_SBGR.Rda", "arr", "arr")
dep <- aggregate_actual("TATIC/Departures/dataTF_dep_SBGR.Rda", "dep", "dep")
sch_arr <- aggregate_schedule("SIROS/Arrivals/dataSR_arr_SBGR.Rda",
                              "Chegada Prevista", "scheduled_arrivals")
sch_dep <- aggregate_schedule("SIROS/Departures/dataSR_dep_SBGR.Rda",
                              "Partida Prevista", "scheduled_departures")

grid <- data.table(time = seq(start_time, end_time, by = "15 min"))
for (x in list(arr, dep, sch_arr, sch_dep)) grid <- merge(grid, x, by = "time", all.x = TRUE)

count_cols <- setdiff(names(grid), "time")
for (nm in count_cols) set(grid, which(is.na(grid[[nm]])), nm, 0L)

grid[, total_n10 := arr_n10 + dep_n10]
grid[, total_n28 := arr_n28 + dep_n28]
grid[, mixed_directions := total_n10 > 0L & total_n28 > 0L]
grid[, observed_direction := fifelse(total_n10 > 0L & total_n28 == 0L, "10",
                              fifelse(total_n28 > 0L & total_n10 == 0L, "28", NA_character_))]
grid[, label_source := fifelse(mixed_directions, "mixed",
                        fifelse(!is.na(observed_direction), "observed", "no_operations"))]
grid[, observed_configuration := fifelse(observed_direction == "10", "10R|10L",
                                   fifelse(observed_direction == "28", "28L|28R", NA_character_))]

# The active state between movements is conservatively carried forward from the
# latest unambiguous operational evidence. Mixed-direction bins remain excluded
# as target labels but do not erase the preceding state.
locf_character <- function(x) {
  observed <- which(!is.na(x))
  out <- rep(NA_character_, length(x))
  if (length(observed)) {
    lookup <- findInterval(seq_along(x), observed)
    valid <- lookup > 0L
    out[valid] <- x[observed[lookup[valid]]]
  }
  out
}
grid[, state_configuration := locf_character(observed_configuration)]
grid[label_source == "no_operations" & !is.na(state_configuration),
     primary_label_source := "carried_forward_no_operations"]
grid[label_source == "observed", primary_label_source := "observed"]
grid[label_source == "mixed", primary_label_source := "mixed_excluded"]
grid[, primary_configuration := fifelse(label_source == "mixed", NA_character_,
                                 state_configuration)]

# METAR values are joined backward in time, never from a future report.
metar <- load_object("METAR/dataMETAR_SBGR.Rda", "dataMETAR")
metar[, observation_time := as.POSIXct(time, tz = "UTC")]
metar <- metar[observation_time <= end_time]
setorder(metar, observation_time)
metar <- metar[, .SD[.N], by = observation_time]
metar_cols <- setdiff(names(metar), "time")
metar <- metar[, ..metar_cols]
metar[, join_time := observation_time]
grid[, issue_time := time]
setkey(metar, join_time)
setkey(grid, time)
grid <- metar[grid, on = .(join_time = time), roll = Inf]
setnames(grid, "observation_time", "metar_time")
grid[, time := issue_time]
grid[, c("join_time", "issue_time") := NULL]
grid[, metar_age_minutes := as.numeric(difftime(time, metar_time, units = "mins"))]

setorder(grid, time)
fwrite(grid, out, dateTimeAs = "ISO")
