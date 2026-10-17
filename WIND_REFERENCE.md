# Wind-direction reference convention

The runway-relative wind features in this repository use a **true-north reference frame**.

METAR wind direction is treated as a meteorological direction referenced to true north. The analysis therefore projects the wind vector onto the SBGR runway axis using true runway bearings:

- runway 10 direction: **73.69° true**;
- reciprocal runway 28 direction: **253.67° true**.

These values are the runway-axis bearings used by the forecasting analysis. They are distinct from the runway designators 10/28 and from the corresponding approximate magnetic headings. Keeping both the meteorological wind direction and runway bearing in the same true-north frame avoids a magnetic/true-north reference mismatch.

The convention is implemented explicitly in `code/run_analysis.py`, including the runway-relative wind features used by the forecasting models and wind-rule comparator.

Related earlier SBGR runway-capacity and tactical ATFM studies used the same convention with the runway bearings rounded to 73°/253°. The greater precision in this repository reflects the bearing values adopted for this study and does not imply a different reference frame.

This file documents the existing implementation only; it does not modify the analysis code, input data, archived results, or published findings.
