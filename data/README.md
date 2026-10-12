# Data

The full refit uses five R data files for São Paulo/Guarulhos (SBGR), January 1, 2022 to
June 30, 2024. `code/prepare_data.R` reads exactly these five files and nothing else.
The three METAR/SIROS objects derive from public sources; the two TATIC objects are
restricted and must never be redistributed. The artifact-only reproduction path does
not require any of these source files.

| File | R object | Rows | Source | Status |
|---|---|---:|---|---|
| `METAR/dataMETAR_SBGR.Rda` | `dataMETAR` | 27,717 | Historical METAR reports, REDEMET (DECEA) | Public-source processed input |
| `SIROS/Arrivals/dataSR_arr_SBGR.Rda` | `dataSR` | 312,094 | Planned arrivals, ANAC air services registry (SIROS) | Public-source processed input |
| `SIROS/Departures/dataSR_dep_SBGR.Rda` | `dataSR` | 310,129 | Planned departures, ANAC air services registry (SIROS) | Public-source processed input |
| `TATIC/Arrivals/dataTF_arr_SBGR.Rda` | `dataTF` | 405,949 | Actual landings and runways, TATIC TWR at TWR-GRU | **Restricted; not in the repository** |
| `TATIC/Departures/dataTF_dep_SBGR.Rda` | `dataTF` | 405,816 | Actual takeoffs and runways, TATIC TWR at TWR-GRU | **Restricted; not in the repository** |

MD5 checksums of the files used for the reported results:

```
593ed37590fb9b91055e9aa0dad9872a  METAR/dataMETAR_SBGR.Rda
49f9c6597bf4b44b34a71b8f947a2128  SIROS/Arrivals/dataSR_arr_SBGR.Rda
95b34e582416720fb23d0cf72d1fc3cb  SIROS/Departures/dataSR_dep_SBGR.Rda
947e6b0ef5a832a102df31628cba92cd  TATIC/Arrivals/dataTF_arr_SBGR.Rda
06d17f706151c1222b8a16d02ed66809  TATIC/Departures/dataTF_dep_SBGR.Rda
```

## Columns used

- **METAR** (`dataMETAR`): `time` (UTC report time), `wind_speed`, `wind_direction`,
  `gust_speed` (kt, degrees true), `visibility`, `ceiling`, `temperature`,
  `dew_point`, `relative_humidity`, `pressure`, and the `ifr`, `lifr`, `ts`
  indicators. Reports are joined backward in time to each 15-minute issue time, with
  a maximum accepted age of 120 minutes.
- **SIROS** (`dataSR`): `Chegada Prevista` (planned arrival time) and
  `Partida Prevista` (planned departure time), counted per 15-minute target bin.
- **TATIC TWR** (`dataTF`): `arr` / `dep` (actual landing / takeoff time) and
  `runway`. Runways beginning with 09 or 10 are grouped as direction 10, and those
  beginning with 27 or 28 as direction 28.

The METAR and SIROS objects were extracted from the public sources and stored as R
data files before this analysis. The scripts used for that extraction are not part
of this repository. Their exact author-bundle copies, row counts, and checksums are
documented in `../PROVENANCE.md`.

## Restricted data

The TATIC TWR records are operational extracts from the Guarulhos control tower.
They are not publicly downloadable and cannot be redistributed by the authors.
NAV Brasil provides the air traffic control service at TWR-GRU. Access to these
records requires a separate request to NAV Brasil through its institutional contact
channels and is subject to its authorization and to applicable operational and
information-security restrictions. The authors cannot guarantee that access will be
granted.

Researchers who obtain the records should place them at the paths above. They are
excluded from version control by `.gitignore`. Without them, the full model-fitting
pipeline cannot run, but every reported table, macro, and figure can still be
regenerated from the archived results in `../results/`.
