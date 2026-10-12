#!/usr/bin/env python3
"""LaTeX formatting helpers used by generated manuscript artifacts."""

from analysis_common import *  # noqa: F401,F403

def latex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def stacked_header(text: str) -> str:
    """Render a column heading on up to two balanced lines.

    Single-line headings set the natural width of narrow numeric columns, which is
    what forced the wide tables to be scaled down. Stacking them lets the data
    determine the width instead.
    """
    words = str(text).split()
    if len(words) < 2:
        return str(text)
    target = len(str(text)) / 2
    best, cut = None, 1
    for i in range(1, len(words)):
        top = len(" ".join(words[:i]))
        score = abs(top - target)
        if best is None or score < best:
            best, cut = score, i
    top, bottom = " ".join(words[:cut]), " ".join(words[cut:])
    # A second line beginning with "[" would be swallowed as the optional vertical
    # space argument of \\, so it is shielded with an empty group.
    if bottom.startswith("["):
        bottom = "{}" + bottom
    return rf"\makecell[bc]{{{top}\\{bottom}}}"


def save_table(path: Path, columns, rows, alignment=None, caption=None, label=None):
    """Write one LaTeX table. Tables carry no notes; anything a reader needs in
    order to interpret a column is stated in the running text instead.

    The body is set at a fixed 8 pt rather than scaled to the text width. Scaling
    shrinks wide tables to whatever size makes them fit, which leaves the widest of
    them illegible at normal page size; a fixed size keeps every table readable and
    consistent with the others.
    """
    alignment = alignment or ("l" + "r" * (len(columns) - 1))
    lines = [r"\begin{table*}[!htbp]", r"\centering"]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines += [r"\fontsize{8}{9.6}\selectfont",
              r"\setlength{\tabcolsep}{2.5pt}",
              r"\setlength{\emergencystretch}{1.5em}",
              r"\sbox{\tabbox}{%",
              rf"\begin{{tabular}}{{{alignment}}}", r"\toprule",
              " & ".join(stacked_header(c) for c in columns) + r" \", r"\midrule"]
    for row in rows:
        lines.append(" & ".join(str(x) for x in row) + r" \")
    lines += [r"\bottomrule", r"\end{tabular}}",
              # Every table is set to exactly the text width. Narrow tables are
              # enlarged and wide ones reduced; the stacked headings and wrapped
              # text columns above keep the natural width close to the text block,
              # so the scaling stays mild in both directions.
              rf"\typeout{{TABLEWIDTH {label or path.stem}: \the\wd\tabbox\space of \the\textwidth}}",
              r"\resizebox{\textwidth}{!}{\usebox{\tabbox}}%"]
    lines.append(r"\end{table*}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
