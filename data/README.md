# Data provenance and local file layout

The full model-fitting pipeline reads five R data files for São Paulo/Guarulhos
(SBGR), covering 1 January 2022 through 30 June 2024. The source-data copies used
for the reported analysis are **not redistributed in this public repository**.
`code/prepare_data.R` reads exactly the five paths below.

| Expected local path | R object | Rows | Source | Availability |
|---|---|---:|---|---|
| `METAR/dataMETAR_SBGR.Rda` | `dataMETAR` | 27,717 | Historical METAR reports, REDEMET (DECEA) | Publicly obtainable; processed copy not republished |
| `SIROS/Arrivals/dataSR_arr_SBGR.Rda` | `dataSR` | 312,094 | Planned arrivals, ANAC air services registry (SIROS) | Publicly obtainable; processed copy not republished |
| `SIROS/Departures/dataSR_dep_SBGR.Rda` | `dataSR` | 310,129 | Planned departures, ANAC air services registry (SIROS) | Publicly obtainable; processed copy not republished |
| `TATIC/Arrivals/dataTF_arr_SBGR.Rda` | `dataTF` | 405,949 | Actual landings and runways, TATIC TWR at TWR-GRU | **Restricted** |
| `TATIC/Departures/dataTF_dep_SBGR.Rda` | `dataTF` | 405,816 | Actual takeoffs and runways, TATIC TWR at TWR-GRU | **Restricted** |

MD5 checksums of the exact files used for the reported results:

```text
593ed37590fb9b91055e9aa0dad9872a  METAR/dataMETAR_SBGR.Rda
49f9c6597bf4b44b34a71b8f947a2128  SIROS/Arrivals/dataSR_arr_SBGR.Rda
95b34e582416720fb23d0cf72d1fc3cb  SIROS/Departures/dataSR_dep_SBGR.Rda
947e6b0ef5a832a102df31628cba92cd  TATIC/Arrivals/dataTF_arr_SBGR.Rda
06d17f706151c1222b8a16d02ed66809  TATIC/Departures/dataTF_dep_SBGR.Rda
```

SHA-256 checksums are recorded in `../PROVENANCE.md`.

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

The METAR and SIROS objects were extracted from their public source systems and
stored as R data files before this analysis. The scripts used for that extraction
are not part of the recovered reproducibility package, so this repository documents
the exact processed-file checksums rather than claiming end-to-end reconstruction
from provider downloads.

## Restricted TATIC records

The TATIC TWR records are operational extracts from the Guarulhos control tower
(TWR-GRU). They are not publicly downloadable and cannot be redistributed by the
authors. Access requires a separate request to NAV Brasil through its institutional
contact channels and remains subject to authorization and applicable operational
and information-security restrictions. The authors cannot guarantee access.

Researchers with legitimate access should recreate the directory structure above
and place authorized local copies there. All source-data files are ignored by Git
so they cannot be committed accidentally. Without the five source files, the full
model-fitting pipeline cannot run; however, every reported table, prose macro, and
figure can still be regenerated from the archived records in `../results/`.
