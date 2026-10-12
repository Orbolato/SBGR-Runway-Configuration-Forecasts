# Restricted: TATIC TWR movement records

This folder must contain the two restricted files used by `code/prepare_data.R`:

```
data/TATIC/Arrivals/dataTF_arr_SBGR.Rda     (R object `dataTF`, 405,949 rows)
data/TATIC/Departures/dataTF_dep_SBGR.Rda   (R object `dataTF`, 405,816 rows)
```

They are operational extracts of TATIC TWR at the Guarulhos control tower
(TWR-GRU) and cannot be redistributed. They are excluded from version control by
`.gitignore`. Access requires a request to NAV Brasil; see `data/README.md`.

Without them, all tables, macros, and figures can still be regenerated from the
archived results in `results/` (see the main README).
