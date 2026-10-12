# Provenance

This repository was initialized from the author-supplied reproducibility bundle
`sbgr-runway-configuration-forecasts.zip`, received on October 12, 2026.

- Bundle SHA-256: `f902e43e24e52396a44bf3d382d3ec9cdaec2486c812ed7ea606e631e4afd219`
- Manuscript: JATM-D-26-00770
- Associated article: *When Do Data-Driven Models Improve Runway-Configuration Forecasts? Multi-Horizon Operational Evidence from São Paulo/Guarulhos*
- Venue: *Journal of Air Transport Management*, accepted for publication (2026)
- Random seed used for reported results: `20260720`

## Public-source processed inputs preserved in this repository

The author-supplied bundle contained the following processed inputs derived from
public sources. The exact author-bundle copies used in the analysis are preserved
in this repository at the paths below. Their row counts and SHA-256 checksums are
recorded so the committed bytes can be verified independently.

| File | Rows | SHA-256 |
| --- | ---: | --- |
| `data/METAR/dataMETAR_SBGR.Rda` | 27,717 | `eae98ebe04dc4a072e1d718b35236367777ed7ebffcb9541c881cdd3654a5f77` |
| `data/SIROS/Arrivals/dataSR_arr_SBGR.Rda` | 312,094 | `051894385eebc639d07422df02fc13b21acbc4e5aad38c08e2ac3954860a2ba4` |
| `data/SIROS/Departures/dataSR_dep_SBGR.Rda` | 310,129 | `f5d94fd7d8e52addb1a4c728e01fe9e7a4456718867bdca184c400914def0bf0` |

These are processed analysis inputs rather than raw-source downloads. The scripts
that originally extracted them from REDEMET and SIROS are not part of the bundle or
this repository. No separate data license is asserted here; applicable third-party
source terms remain in force.

The archived numerical records used by the public artifact-only reproduction path
are also preserved exactly:

| File | SHA-256 |
| --- | --- |
| `results/analysis_results.json` | `90c8792270694f222095578dd0c7621171e5387f546beb157549dbb91c890bad` |
| `results/figure_data.json` | `92a1f86c8b648e2b054a40d6a5804a98feede42846bbdcc4348bbd0e85b148a6` |
| `results/figure_source_values.json` | `723526b253b01b7554ca6df31c0e716172efc3882e085463ed45a78561259001` |

## Restricted files deliberately excluded

The internal author bundle also contained the two TATIC TWR files used for the
full analysis. Their presence in that private/internal bundle does **not** imply
permission to redistribute them. They are intentionally absent from the public
GitHub repository.

| Expected path | Rows | MD5 used in the study | SHA-256 of author-held copy |
| --- | ---: | --- | --- |
| `data/TATIC/Arrivals/dataTF_arr_SBGR.Rda` | 405,949 | `947e6b0ef5a832a102df31628cba92cd` | `decf90eebfeb4aa19a6d2ff3d147a3456759e3b612c4921fa72ffbd3cf40e333` |
| `data/TATIC/Departures/dataTF_dep_SBGR.Rda` | 405,816 | `06d17f706151c1222b8a16d02ed66809` | `80b6ffc747251afc60a20b5bbd45b93ad66154311dd800632b9741ec997b4b4c` |

Access requires separate authorization from NAV Brasil, as described in
`data/README.md`. The `.gitignore` rules are intentionally defensive so that local
authorized copies and any derived interval grid cannot be committed accidentally.

## Reproduction claim

The public repository supports exact regeneration of Tables 3-10,
`generated_results.tex`, and Figures 1-7 from the archived result records without
restricted operational data. It does **not** claim that a user without TATIC access
can refit the forecasting models from the source records.

A full rerun with separately authorized TATIC data and the three included
public-source processed inputs can additionally be compared against the archived
results using `code/verify_outputs.py`; wall-clock timing fields are reported
separately because they are machine dependent.
