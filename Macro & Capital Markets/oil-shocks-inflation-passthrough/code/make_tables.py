"""
Converts the results workbook produced by oil_passthrough_analysis.py into LaTeX tables for the paper.

Usage:  python code/make_tables.py output/results_tables.xlsx latex/tables
"""
import pandas as pd, numpy as np, sys
X = pd.read_excel(sys.argv[1], sheet_name=None)
T = sys.argv[2]
def f(x, d=2):
    if pd.isna(x): return "--"
    s_ = f"{x:.{d}f}"
    if float(s_) == 0: s_ = s_.lstrip("-")
    return s_.replace("-", "$-$")
def stars(b, se):
    t = abs(b / se) if se and not pd.isna(se) else 0
    return "^{***}" if t > 2.576 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""
def bse(b, se, d=2): return f"${f(b,d).replace('$','')}{stars(b,se)}$ \\newline {{\\footnotesize ({f(se,d)})}}" if False else f"${f(b,d).replace('$','')}{stars(b,se)}$"
def sep(se, d=2): return f"{{\\footnotesize ({f(se,d)})}}"
def w(name, s): open(f"{T}/{name}.tex", "w").write(s)

# ---------- Table: pass-through
A = X["A_passthrough"]
rows = []
for ser in ["US headline CPI", "US core CPI", "Euro area headline HICP", "Euro area core HICP"]:
    rows.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{ser}}}}}\\\\")
    for smp in ["Full sample", "Pre-2020", "2020-2026"]:
        r = A[(A.series == ser) & (A["sample"] == smp)].iloc[0]
        rows.append(f"\\quad {smp.replace('-', '--')} & " + " & ".join(f(r[f'h={h}']) for h in [0, 3, 6, 12, 18, 24]) + f" & {int(r.obs)}\\\\")
    rows.append("\\addlinespace")
w("tab_passthrough", r"""\begin{tabular}{lrrrrrrr}
\toprule
 & \multicolumn{6}{c}{Months after the oil move ($h$)} & \\ \cmidrule(lr){2-7}
 & 0 & 3 & 6 & 12 & 18 & 24 & Obs.\\
\midrule
""" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}")

# ---------- Table: market betas by period
B = X["B_market_betas"]
ser = ["US 5y breakeven", "US 5y5y swap", "EA 5y swap", "EA 5y5y swap"]
rows = []
for p in B.period.unique():
    r = [B[(B.period == p) & (B.series == s)].iloc[0] for s in ser]
    lab = p.replace("Iran shock (Mar-Nov 2026)", "Iran shock (Mar--Nov 2026)").replace("-", "--") if "Iran" not in p else "Iran shock (Mar--Nov 2026)"
    rows.append(f"{lab} & " + " & ".join(bse(x['bp per +10% Brent'], x['s.e.']) for x in r) + f" & {int(r[0].weeks)}\\\\")
    rows.append(" & " + " & ".join(sep(x['s.e.']) for x in r) + " & \\\\")
    if p == "2004-2026": rows.append("\\midrule")
w("tab_market", r"""\begin{tabular}{lrrrrr}
\toprule
 & \multicolumn{2}{c}{United States} & \multicolumn{2}{c}{Euro area} & \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}
Period & 5y BEI & 5y5y swap & 5y swap & 5y5y swap & Weeks\\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# ---------- Table: by shock type (market)
F = X["F1_market_by_shock_type"]
rows = []
for p in ["2004-2026", "2004-2019", "2020-2026"]:
    rows.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{p.replace('-', '--')}}}}}\\\\")
    for k, lab, sk in [("demand-driven: bp per +10%", "Global-demand-driven oil rise", "s.e. demand"), ("oil-specific: bp per +10%", "Oil-specific (supply) rise", "s.e. oil-specific")]:
        r = [F[(F.period == p) & (F.series == s)].iloc[0] for s in ser]
        rows.append(f"\\quad {lab} & " + " & ".join(bse(x[k], x[sk]) for x in r) + "\\\\")
        rows.append(" & " + " & ".join(sep(x[sk]) for x in r) + "\\\\")
    rows.append("\\addlinespace")
F3 = X["F3_identification_robustness"]
rows.append("\\midrule\n\\multicolumn{5}{l}{\\textit{Range across admissible identifications, 2004--2026}}\\\\")
for k, lab in [("demand-driven: bp per +10%", "Global-demand-driven"), ("oil-specific: bp per +10%", "Oil-specific (supply)")]:
    rr = []
    for s in ser:
        v = F3[(F3.series == s) & (F3.identification != "baseline (midpoint)")][k]
        rr.append(f"[{f(v.min(),1)}, {f(v.max(),1)}]")
    rows.append(f"\\quad {lab} & " + " & ".join(rr) + "\\\\")
w("tab_shocktype", r"""\begin{tabular}{lrrrr}
\toprule
 & \multicolumn{2}{c}{United States} & \multicolumn{2}{c}{Euro area} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}
 & 5y BEI & 5y5y swap & 5y swap & 5y5y swap\\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# ---------- Table: inflation by shock type (compact)
F2 = X["F2_inflation_by_shock_type"]
rows = []
for s in ["US headline CPI", "US core CPI", "Euro area headline HICP", "Euro area core HICP"]:
    rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{s}}}}}\\\\")
    for k, lab in [("demand-driven", "Global-demand-driven"), ("oil-specific", "Oil-specific (supply)"), ("all oil moves (same sample)", "All oil moves")]:
        r = F2[(F2.series == s) & (F2.shock == k)].iloc[0]
        rows.append(f"\\quad {lab} & " + " & ".join(f(r[f'h={h}']) for h in [3, 6, 12, 18, 24]) + "\\\\")
    rows.append("\\addlinespace")
w("tab_infl_shocktype", r"""\begin{tabular}{lrrrrr}
\toprule
 & \multicolumn{5}{c}{Months after the oil move ($h$)}\\ \cmidrule(lr){2-6}
 & 3 & 6 & 12 & 18 & 24\\
\midrule
""" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}")

# ---------- Table: out of sample
E = X["E_out_of_sample_2026"]
rows = []
for s in ["US headline CPI", "US core CPI", "Euro area headline HICP", "Euro area core HICP"]:
    for m, lab in [("1990-2019 model", "1990--2019"), ("model to Feb 2026", "1990--Feb 2026")]:
        r = E[(E.series == s) & (E.model == m)].iloc[0]
        first = s if m == "1990-2019 model" else ""
        inband = f"{100*r['share of months inside 90% band']:.0f}\\%" if not pd.isna(r['share of months inside 90% band']) else "--"
        rows.append(f"{first} & {lab} & {f(r['actual at end'],1)} & {f(r['forecast at end'])} & {f(r['no-shock counterfactual at end'])} & {f(r['oil contribution at end (pp)'])} & {f(r['RMSE'])} & {f(r['RMSE no-shock counterfactual'])} & {inband}\\\\")
    rows.append("\\addlinespace")
w("tab_oos", r"""\begin{tabular}{llrrrrrrr}
\toprule
 & Estimation & Actual & Forecast & No-shock & Oil & RMSE & RMSE & In 90\%\\
Series & sample & (end) & (end) & (end) & contrib. & model & no-shock & band\\
\midrule
""" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}")

# ---------- Table: regime & asymmetry
G = X["G1_inflation_regime_asymmetry"]
rows = []
cases = [("inflation <= 3% (last month)", "Inflation $\\leq$ 3\\%"), ("inflation > 3% (last month)", "Inflation $>$ 3\\%"),
         ("oil rise (+10%)", "Oil rise"), ("oil fall (effect of -10%, sign flipped)", "Oil fall (sign flipped)")]
for s in ["US headline CPI", "US core CPI", "Euro area headline HICP", "Euro area core HICP"]:
    r = {c: G[(G.series == s) & (G.case == c)].iloc[0] for c, _ in cases}
    rows.append(f"{s} & " + " & ".join(f"{f(r[c]['h=6'])} & {f(r[c]['h=12'])}" for c, _ in cases) + "\\\\")
w("tab_regime", r"""\begin{tabular}{lrrrrrrrr}
\toprule
 & \multicolumn{2}{c}{Inflation $\leq$ 3\%} & \multicolumn{2}{c}{Inflation $>$ 3\%} & \multicolumn{2}{c}{Oil rise} & \multicolumn{2}{c}{Oil fall (flipped)}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
 & $h=6$ & $h=12$ & $h=6$ & $h=12$ & $h=6$ & $h=12$ & $h=6$ & $h=12$\\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
G2 = X["G2_market_regime_asymmetry"]
rows = []
for s in ser:
    r = G2[G2.series == s].iloc[0]
    rows.append(f"{s.replace('EA','Euro area')} & {bse(r['low-inflation regime'], r['s.e. low'])} & {bse(r['high-inflation regime'], r['s.e. high'])} & {bse(r['oil up'], r['s.e. up'])} & {bse(r['oil down'], r['s.e. down'])}\\\\")
    rows.append(f" & {sep(r['s.e. low'])} & {sep(r['s.e. high'])} & {sep(r['s.e. up'])} & {sep(r['s.e. down'])}\\\\")
w("tab_regime_mkt", r"""\begin{tabular}{lrrrr}
\toprule
 & Inflation $\leq$ 3\% & Inflation $>$ 3\% & Oil up & Oil down\\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# ---------- Table: robustness
H = X["H1_robustness_long"]
rows = []
for sp in H.specification.unique():
    r = [H[(H.specification == sp) & (H.series == s)].iloc[0] for s in ["US headline CPI", "US core CPI", "Euro area headline HICP", "Euro area core HICP"]]
    lab = sp.replace("Excl.", "Excluding").replace("2008-09", "2008--09").replace("2004-2026", "2004--2026").replace("1990-Feb", "1990--Feb")
    rows.append(f"{lab} & " + " & ".join(f"{f(x['h=6'])} & {f(x['h=12'])}" for x in r) + "\\\\")
    if sp.startswith("Baseline"): rows.append("\\midrule")
w("tab_robust", r"""\begin{tabular}{lrrrrrrrr}
\toprule
 & \multicolumn{2}{c}{US headline} & \multicolumn{2}{c}{US core} & \multicolumn{2}{c}{EA headline} & \multicolumn{2}{c}{EA core}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Specification & $h=6$ & $h=12$ & $h=6$ & $h=12$ & $h=6$ & $h=12$ & $h=6$ & $h=12$\\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
H2 = X["H2_robustness_market_frequency"]
rows = []
for s in ser:
    r = H2[H2.series == s].iloc[0]
    rows.append(f"{s.replace('EA','Euro area')} & {f(r['daily (same day)'])} & {bse(r['daily (lead+same+lag)'], r['s.e. daily sum'])} & {bse(r['weekly (baseline)'], r['s.e. weekly'])} & {bse(r['monthly'], r['s.e. monthly'])}\\\\")
    rows.append(f" & & {sep(r['s.e. daily sum'])} & {sep(r['s.e. weekly'])} & {sep(r['s.e. monthly'])}\\\\")
w("tab_robust_mkt", r"""\begin{tabular}{lrrrr}
\toprule
 & Daily & Daily & Weekly & Monthly\\
 & (same day) & ($t-1$ to $t+1$) & (baseline) & \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# ---------- Table: consistency + episodes
C = X["C_consistency"]; Dp = X["D_2022_vs_2026"]
rows = []
for _, r in C.iterrows():
    rows.append(f"{r.region} & {f(r['price-level effect after 12m (%)'])} & {f(r['after 24m (%)'])} & {f(r['implied rise in 5y average inflation (bp)'],1)} & {f(r['observed 5y market move (bp)'],1)} & {f(r['market / implied'])}\\\\")
w("tab_consistency", r"""\begin{tabular}{lrrrrr}
\toprule
 & \multicolumn{2}{c}{Price-level effect (\%)} & Implied 5y avg. & Observed 5y & \\ \cmidrule(lr){2-3}
Region & 12 months & 24 months & inflation (bp) & market move (bp) & Ratio\\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
rows = []
for _, r in Dp.iterrows():
    rows.append(f"{r.episode.replace('/', '/ ')} & {f(r['Brent peak move (%)'],1)} & " + " & ".join(f(r[f'{s} at Brent peak (bp)'],1) for s in ser) + "\\\\")
    rows.append("\\quad max.\\ move within 120 days & & " + " & ".join(f(r[f'{s} max move in 120 days (bp)'],1) for s in ser) + "\\\\")
w("tab_episodes", r"""\begin{tabular}{lrrrrr}
\toprule
 & Brent & \multicolumn{2}{c}{United States (bp)} & \multicolumn{2}{c}{Euro area (bp)}\\ \cmidrule(lr){3-4}\cmidrule(lr){5-6}
Episode & peak (\%) & 5y BEI & 5y5y & 5y & 5y5y\\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
print("ok")
