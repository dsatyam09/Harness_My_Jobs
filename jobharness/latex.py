"""Write booktabs tables that the progress report pulls in with LaTeX's input command.

Tables are wrapped in adjustbox (max width = line width), so the report must load adjustbox."""
from __future__ import annotations


def esc(s) -> str:
    s = str(s)
    for a, b in [("\\", "\\textbackslash{}"), ("_", "\\_"), ("&", "\\&"), ("#", "\\#")]:
        s = s.replace(a, b)
    return s


def write_table(headers, rows, colspec: str, note: str | None = None) -> str:
    lines = ["{\\small", "\\begin{adjustbox}{max width=\\linewidth}",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
             " & ".join(headers) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append(" & ".join(str(c) for c in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    if note:
        lines += ["", "\\smallskip", f"{{\\footnotesize {note}\\par}}"]
    lines.append("}")
    return "\n".join(lines) + "\n"
