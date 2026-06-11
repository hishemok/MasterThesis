import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "src" / "braiding"
OUTPUT_PATH = ROOT / "texmex" / "generated" / "matched_operator_coefficients.tex"

INTERACTIONS = ("0.0", "0.1", "0.5", "1.0", "2.0")

RECORD_PATTERN = re.compile(
    r"(?ms)^(\d+)\t(\d+)\t[^\t]+\t[^\t]+\t[^\t]+\t[^\t]+\t[^\t]+\s*\t"
    r"(.*?)\t(.*?)\t([0-9.]+)\n(?=\d+\t|\Z)"
)
COMPLEX_PATTERN = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*[+-]\s*(?:\d+(?:\.\d*)?|\.\d+)j"
)


def parse_gamma2_coefficients(raw_array):
    values = [complex(value.replace(" ", "")) for value in COMPLEX_PATTERN.findall(raw_array)]
    if len(values) % 2:
        raise ValueError(f"Expected coefficient pairs, found {len(values)} values.")
    return [value.real for value in values[1::2]]


def format_coefficients(values):
    return r"\([" + r",\allowbreak\ ".join(f"{value:+.2f}" for value in values) + r"]\)"


def format_sign_products(b_values, c_values):
    if len(b_values) != len(c_values):
        raise ValueError("B and C coefficient arrays contain different numbers of blocks.")
    products = ["+" if b_value * c_value >= 0 else "-" for b_value, c_value in zip(b_values, c_values)]
    return r"\([" + r",\allowbreak\ ".join(products) + r"]\)"


def parse_results(interaction):
    path = RESULT_DIR / f"braiding_results_matched_ops_U={interaction}.txt"
    text = path.read_text()
    rows = []

    for match in RECORD_PATTERN.finditer(text):
        sector = int(match.group(1))
        dimension = int(match.group(2))
        b_values = parse_gamma2_coefficients(match.group(3))
        c_values = parse_gamma2_coefficients(match.group(4))
        rows.append((interaction, sector, dimension, b_values, c_values))

    if not rows:
        raise ValueError(f"No coefficient records parsed from {path}.")
    return rows


def render_table(rows):
    lines = [
        r"\begin{landscape}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{@{}ccc >{\raggedright\arraybackslash}p{6.1cm} >{\raggedright\arraybackslash}p{6.1cm} >{\raggedright\arraybackslash}p{4.2cm}@{}}",
        r"\caption{Blockwise coefficients of the projected local endpoint operators in the energy-level Majorana basis. The entries list the fitted coefficient multiplying the second energy-level Majorana in each labelled block, in the block order used by the fit. The fitted coefficients of the first Majorana component vanish at the saved precision. The final column gives the sign of the product \(b_{B,r}b_{C,r}\), which determines whether the two local operators select the same (\(+\)) or opposite (\(-\)) relative orientation in block \(r\).}",
        r"\label{tab:matched_operator_coefficients}\\",
        r"\toprule",
        r"\(U/t\) & Sector & Dim. & \(b_{B,r}\) & \(b_{C,r}\) & \(\operatorname{sgn}(b_{B,r}b_{C,r})\) \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{6}{c}{\tablename\ \thetable\ continued} \\",
        r"\toprule",
        r"\(U/t\) & Sector & Dim. & \(b_{B,r}\) & \(b_{C,r}\) & \(\operatorname{sgn}(b_{B,r}b_{C,r})\) \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{6}{r}{Continued on next page} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    previous_interaction = None
    for interaction, sector, dimension, b_values, c_values in rows:
        if previous_interaction is not None and interaction != previous_interaction:
            lines.append(r"\addlinespace")
        lines.append(
            f"{interaction} & {sector} & {dimension} & "
            f"{format_coefficients(b_values)} & {format_coefficients(c_values)} & "
            f"{format_sign_products(b_values, c_values)} \\\\"
        )
        previous_interaction = interaction

    lines.extend([r"\end{longtable}", r"\end{landscape}", ""])
    return "\n".join(lines)


def main():
    rows = []
    for interaction in INTERACTIONS:
        rows.extend(parse_results(interaction))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render_table(rows))
    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
