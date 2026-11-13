"""Run the analysis for one or more commodities: data -> panel -> strategies -> tables/figures.

Usage (from the project root):
    python src/run_all.py --commodity CL               # reads data/raw/CL.xlsx
    python src/run_all.py --commodity CL GC HG C       # several, one after the other
    python src/run_all.py --data PATH.xlsx             # every sheet; commodity read from cell A1
    python src/run_all.py --data PATH.xlsx --commodity GC C   # only these sheets
    python src/run_all.py --list                       # supported commodities

Outputs for commodity CODE go to data/processed/CODE/ and results/CODE/
(tables/, figures/, numbers.tex with the key numbers quoted in the paper).
"""
import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd

import config as C
from evaluation import (bootstrap_sharpe_diff, metrics, newey_west_mean,
                        sharpe, spanning, vol_match)
from panel import build_panel
from plots import fig_cumulative, fig_premium, fig_weights
from commodities import INSTRUMENTS, calendar_from_file, contract_calendar, is_spot, uses_smile
from load_data import check_daily, detect_commodity, list_sheets, load_instrument
from rolls import check_calendar, front_returns, spot_returns
from strategies import RULE_LABELS, RULE_SHORT, managed_returns, rule_weights

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------- helpers
def latex_table(df, path, fmt="{:.2f}", index_name=""):
    """Minimal booktabs table writer (no jinja2 dependency)."""
    cols = list(df.columns)
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             " & ".join([index_name] + [str(c) for c in cols]) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        cells = []
        for v in row.values:
            if isinstance(v, str):
                cells.append(v)
            elif pd.isna(v):
                cells.append("--")
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v:d}")
            else:
                cells.append(fmt.format(v))
        lines.append(" & ".join([str(idx)] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    text = re.sub(r"(?<!\\)%", r"\\%", "\n".join(lines))   # escape bare % signs
    path.write_text(text)


def run_pipeline(daily, calendar, roll_days=C.ROLL_DAYS_BEFORE_EXPIRY, strike=C.STRIKE,
                 code=None, **rule_kw):
    """calendar=None: continuous series (ETF, credit index), no futures roll."""
    if calendar is None:
        front = spot_returns(daily, code)
    else:
        front = front_returns(daily, calendar, roll_days=roll_days)
    panel = build_panel(daily, front, strike=strike)
    w = rule_weights(panel, **rule_kw)
    return daily, calendar, front, panel, w


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--commodity", nargs="+", metavar="CODE",
                    help="commodity codes, e.g. CL GC HG C (default: read from the data file)")
    ap.add_argument("--data", help="Bloomberg Excel export; one commodity per sheet "
                                   f"(default ${C.DATA_ENV_VAR} or data/raw/CODE.xlsx)")
    ap.add_argument("--calendar", help="optional contract calendar CSV from Bloomberg "
                                       "(default data/raw/CODE_calendar.csv if it exists)")
    ap.add_argument("--strike", choices=["smile", "atm"], default=C.STRIKE,
                    help="variance-swap strike: implied variance from the 30-day smile "
                         "(default) or ATM implied variance")
    ap.add_argument("--list", action="store_true", help="list supported commodities and exit")
    args = ap.parse_args(argv)

    if args.list:
        for code, s in INSTRUMENTS.items():
            what = f"months {s['cycle']}" if "cycle" in s else f"continuous ({s['units']})"
            strike = "smile" if s.get("smile", True) else "ATM"
            print(f"{code:<5} {s['name']:<32} {s['exchange']:<9} {what:<20} strike {strike}")
        return

    codes = [c.upper() for c in args.commodity] if args.commodity else None
    for code in codes or []:
        if code not in INSTRUMENTS:
            ap.error(f"unknown instrument {code}; supported: {', '.join(INSTRUMENTS)}")

    jobs = []                                   # (code, file, sheet)
    given = args.data or (os.environ.get(C.DATA_ENV_VAR) if not codes or len(codes) == 1 else None)
    if given:
        # one workbook, one commodity per sheet (detected from cell A1)
        path = C.resolve_data_path(codes[0] if codes and len(codes) == 1 else "?", given)
        sheets = [(name, code) for name, code in list_sheets(path) if code in INSTRUMENTS]
        if codes:
            sheets = [(n, c) for n, c in sheets if c in codes]
            missing = set(codes) - {c for _, c in sheets}
            if missing:
                ap.error(f"{path} has no sheet for {', '.join(sorted(missing))} (cell A1 ticker)")
        if not sheets:
            ap.error(f"no sheet of {path} has a recognised ticker in cell A1; use --commodity")
        # skip unusable sheets; if two sheets hold the same commodity keep the larger
        usable = {}
        for name, code in sheets:
            daily = load_instrument(path, name, code)
            issues = check_daily(daily, f"[{code}] sheet {name}")
            if issues:
                print("SKIPPED " + "; ".join(issues))
                continue
            if code in usable and len(usable[code][1]) >= len(daily):
                print(f"[{code}] sheet {name} ignored: sheet {usable[code][0]} has more data")
                continue
            usable[code] = (name, daily)
        if not usable:
            ap.error(f"no usable sheet in {path}")
        jobs = [(code, path, name) for code, (name, _) in usable.items()]
    elif codes:
        for code in codes:
            path = C.resolve_data_path(code, None, use_env=False)
            match = [n for n, c in list_sheets(path) if c == code]
            jobs.append((code, path, match[0] if match else 0))
    else:
        ap.error("say which commodity to run, e.g. --commodity CL "
                 "(reads data/raw/CL.xlsx), or pass --data PATH")

    if args.calendar and len(jobs) > 1:
        ap.error("--calendar can only be used with a single commodity")
    for code, path, sheet in jobs:
        analyse(code, path, args.calendar, sheet=sheet, strike=args.strike)


def analyse(code, data_path, calendar_path=None, sheet=0, strike=C.STRIKE):
    spec = INSTRUMENTS[code]
    where = f"{data_path}" + (f" [sheet {sheet}]" if sheet != 0 else "")
    print(f"[{code}] {spec['name']}: reading {where}")
    raw_daily = load_instrument(data_path, sheet, code)
    found = detect_commodity(data_path, sheet)
    if found and found != code:
        print(f"[{code}] warning: cell A1 of the file looks like {found}, not {code}")

    cal_path = calendar_path or C.default_calendar_file(code)
    if is_spot(code):
        calendar = None
        print(f"[{code}] continuous series: no futures roll")
    elif calendar_path or cal_path.exists():
        calendar = calendar_from_file(cal_path, raw_daily.index)
        print(f"[{code}] contract calendar from {cal_path}")
    else:
        calendar = contract_calendar(code, raw_daily.index)
        print(f"[{code}] contract calendar from {spec['exchange']} rules")

    out = C.results_dir(code, strike)
    T, F, P = out / "tables", out / "figures", C.processed_dir(code, strike)
    for d in [T, F, P]:
        d.mkdir(parents=True, exist_ok=True)
    macros = {}

    # results are filed under the strike requested; instruments whose smile
    # points are too far out of the money (bond futures, ETFs) always use ATM
    strike_req, strike = strike, (strike if uses_smile(code) else "atm")
    if strike != strike_req:
        print(f"[{code}] strike = ATM implied variance (smile points too far from the money)")
    daily, cal_used, front, p, w = run_pipeline(raw_daily, calendar, strike=strike, code=code)
    if strike == "smile" and p["iv_smile"].isna().all():
        print(f"[{code}] no smile points in the data: strike = ATM implied variance")
    gross = managed_returns(p, w)
    scaled = gross.apply(vol_match)
    labels = [RULE_LABELS[c] for c in gross.columns]

    front.to_csv(P / "daily_front_returns.csv")
    p.to_csv(P / "monthly_panel.csv")
    w.to_csv(P / "weights.csv")
    gross.to_csv(P / "returns_gross_per_vega.csv")
    scaled.to_csv(P / "returns_volmatched.csv")

    # ---- Table A1: calendar validation (futures only)
    if calendar is not None:
        validate_calendar(code, daily, cal_used["last_trade"], T, macros)


    # ---- Table 1: the premium
    rows = {}
    for name, s in [("Implied variance $\\sigma^2_{IV}$", p["iv"] ** 2),
                    ("Realised variance $\\sigma^2_{RV}$", p["rv_next"]),
                    ("Variance premium $\\sigma^2_{IV}-\\sigma^2_{RV}$", p["vrp_var"]),
                    ("Implied vol $\\sigma_{IV}$", p["iv"]),
                    ("Realised vol $\\sigma_{RV}$", np.sqrt(p["rv_next"])),
                    ("Vol premium $\\sigma_{IV}-\\sigma_{RV}$", p["vrp_vol"])]:
        m, t = newey_west_mean(s)
        rows[name] = {"Mean": 100 * m, "t (NW)": t, "Median": 100 * s.median(),
                      "Std": 100 * s.std(), "Share $>0$": (s > 0).mean()}
    t1 = pd.DataFrame(rows).T
    latex_table(t1, T / "tab1_premium.tex", index_name="")
    t1.to_csv(T / "tab1_premium.csv")
    m_var, t_var = newey_west_mean(p["vrp_var"])
    m_vol, t_vol = newey_west_mean(p["vrp_vol"])
    macros.update({
        "NMonths": f"{len(p)}", "SampleStart": p.index[0].strftime("%B %Y"),
        "SampleEnd": p.index[-1].strftime("%B %Y"),
        "MeanIV": f"{100 * p['iv'].mean():.1f}", "MeanRV": f"{100 * np.sqrt(p['rv_next']).mean():.1f}",
        "VolPremium": f"{100 * m_vol:.2f}", "VolPremiumT": f"{t_vol:.2f}",
        "VarPremium": f"{100 * m_var:.2f}", "VarPremiumT": f"{t_var:.2f}",
        "ShareVolPos": f"{100 * (p['vrp_vol'] > 0).mean():.0f}",
        "MedianVolPremium": f"{100 * p['vrp_vol'].median():.2f}",
    })

    # ---- Table 2: performance, vol-matched to 10% p.a.
    t2 = scaled.apply(metrics)
    avg_w = w.mean()
    t2.loc["Average weight"] = avg_w
    t2.loc["Months reduced (%)"] = 100 * (w < 1).mean()
    t2.columns = labels
    t2.to_csv(T / "tab2_performance.csv")
    latex_table(t2.T[["Mean (% p.a.)", "Sharpe", "Sortino", "Skewness", "Max drawdown (%)",
                      "CVaR 95 (%)", "CVaR 99 (%)", "Worst month (%)", "Average weight"]]
                .rename(columns={"Mean (% p.a.)": "Mean", "Max drawdown (%)": "MaxDD",
                                 "CVaR 95 (%)": "CVaR95", "CVaR 99 (%)": "CVaR99",
                                 "Worst month (%)": "Worst", "Average weight": "Avg $w$"}),
                T / "tab2_performance.tex", index_name="Strategy")
    um = t2["Unmanaged"]
    macros.update({
        "UnmSharpe": f"{um['Sharpe']:.2f}", "UnmSkew": f"{um['Skewness']:.2f}",
        "UnmKurt": f"{um['Excess kurtosis']:.1f}", "UnmMDD": f"{um['Max drawdown (%)']:.0f}",
        "UnmWorst": f"{-um['Worst month (%)']:.0f}", "UnmCVaR": f"{um['CVaR 99 (%)']:.0f}",
        "UnmPosShare": f"{100 * (p['ret_short_var'] > 0).mean():.0f}",
        "UnmMedianRet": f"{100 * p['ret_short_var'].median():.1f}",
    })
    best = t2.loc["Sharpe"].drop("Unmanaged").idxmax()
    macros["BestRule"] = best
    macros["BestSharpe"] = f"{t2.loc['Sharpe', best]:.2f}"
    macros["BestMDD"] = f"{t2.loc['Max drawdown (%)', best]:.0f}"
    macros["BestCVaR"] = f"{t2.loc['CVaR 99 (%)', best]:.0f}"

    # ---- Table 3: spanning regressions and Sharpe-difference bootstrap
    rows = {}
    for c in gross.columns.drop("unmanaged"):
        sp = spanning(scaled[c], scaled["unmanaged"])
        d, pval, lo, hi = bootstrap_sharpe_diff(scaled[c], scaled["unmanaged"])
        sp["$\\Delta$SR"], sp["p-value"], sp["95\\% CI"] = d, pval, f"[{lo:.2f}, {hi:.2f}]"
        rows[RULE_LABELS[c]] = sp
    t3 = pd.DataFrame(rows).T
    t3.to_csv(T / "tab3_spanning.csv")
    latex_table(t3[["Alpha (% p.a.)", "t(alpha)", "Beta", "Appraisal ratio", "$\\Delta$SR",
                    "p-value", "95\\% CI"]].rename(columns={"Alpha (% p.a.)": "$\\alpha$",
                                                            "t(alpha)": "$t(\\alpha)$",
                                                            "Beta": "$\\beta$",
                                                            "Appraisal ratio": "AR"}),
                T / "tab3_spanning.tex", index_name="Rule")
    sig = t3[(t3["t(alpha)"].astype(float) > 1.96)]
    macros["NSigAlpha"] = f"{len(sig)}"
    macros["NSigSR"] = f"{int((t3['p-value'].astype(float) < 0.05).sum())}"

    # ---- Table 4: crisis episodes (cumulative vol-matched P&L, %)
    rows = {}
    for name, (a, b) in C.CRISES.items():
        rows[name] = 100 * scaled.loc[a:b].sum()
    t4 = pd.DataFrame(rows).T
    t4.columns = labels
    t4.to_csv(T / "tab4_crises.csv")
    latex_table(t4.T, T / "tab4_crises.tex", fmt="{:.1f}", index_name="Strategy")
    covid = t4.filter(like="COVID", axis=0).iloc[0]
    macros["CovidUnm"] = f"{-covid['Unmanaged']:.0f}"

    # ---- Table 5: transaction costs (Sharpe ratio, net)
    rows = {}
    for cvp in C.TCOST_VOL_POINTS:
        net = managed_returns(p, w, cost_vol_points=cvp)
        rows[f"{cvp:g} vol pt"] = net.apply(sharpe)
    t5 = pd.DataFrame(rows)
    t5.index = labels
    t5.to_csv(T / "tab5_costs.csv")
    latex_table(t5, T / "tab5_costs.tex", index_name="Strategy")

    # ---- Table 6: sub-periods and out-of-sample scaling
    ins = gross.loc[:C.SPLIT_DATE]
    oos = gross.loc[pd.Period(C.SPLIT_DATE, "M") + 1:]
    rows = {}
    for c in gross.columns:
        oos_scaled = oos[c] * C_target_scale(ins[c])
        rows[RULE_LABELS[c]] = {
            "SR in-sample": sharpe(ins[c]), "SR out-of-sample": sharpe(oos[c]),
            "OOS MaxDD (%)": 100 * oos_scaled.cumsum().pipe(lambda s: (s.cummax() - s).max()),
            "OOS CVaR 99 (%)": -100 * oos_scaled[oos_scaled <= oos_scaled.quantile(0.01)].mean(),
            "OOS vol (% p.a.)": 100 * np.sqrt(12) * oos_scaled.std(),
        }
    t6 = pd.DataFrame(rows).T
    t6.to_csv(T / "tab6_subperiods.csv")
    latex_table(t6, T / "tab6_subperiods.tex", index_name="Strategy")
    macros["SplitYear"] = C.SPLIT_DATE[:4]

    # ---- Table 7: robustness (Sharpe ratios of the vol-matched strategies)
    variants = {
        "Baseline": {},
        "Roll at expiry (generic F1)": {"roll_days": 0},
        "Roll 10 days before expiry": {"roll_days": 10},
        "36-month burn-in": {"min_hist": 36},
        "90th percentile thresholds": {"pct": 0.90},
    }
    if calendar is None:                        # no roll to move
        variants = {k: v for k, v in variants.items() if "roll_days" not in v}
    rows = {}
    for name, kw in variants.items():
        _, _, _, pv, wv = run_pipeline(raw_daily, calendar, strike=strike, code=code, **kw)
        rows[name] = managed_returns(pv, wv).apply(sharpe)
    # weight 0 instead of 0.5 for the "reduce" rules
    wz = w.replace(C.REDUCED_WEIGHT, 0.0)
    wz["vol_scaling"] = w["vol_scaling"]
    rows["Reduce = exit (w = 0)"] = managed_returns(p, wz).apply(sharpe)
    # volatility-swap proxy instead of variance swap
    rows["Vol-swap proxy"] = w.mul(p["vrp_vol"], axis=0).apply(sharpe)
    # excluding March-April 2020
    keep = ~p.index.isin(pd.PeriodIndex(["2020-03", "2020-04"], freq="M"))
    rows["Excl. Mar-Apr 2020"] = gross[keep].apply(sharpe)
    t7 = pd.DataFrame(rows).T
    t7.columns = labels
    t7.to_csv(T / "tab7_robustness.csv")
    latex_table(t7.set_axis([RULE_SHORT[c] for c in gross.columns], axis=1),
                T / "tab7_robustness.tex", index_name="Variant")
    macros["UnmSharpeExCovid"] = f"{rows['Excl. Mar-Apr 2020']['unmanaged']:.2f}"

    # ---- Table 8: strike sensitivity. ATM implied vol understates the fair
    # variance-swap strike (convexity / skew), so add x vol points to it
    rows = {}
    for x in [0, 1, 2, 3, 4]:
        k = p["iv"] + x / 100
        r = (k ** 2 - p["rv_next"]) / (2 * k)
        rows[f"+{x} vol pt"] = w.mul(r, axis=0).apply(sharpe)
    t8 = pd.DataFrame(rows)
    t8.index = labels
    t8.to_csv(T / "tab8_strike.csv")
    latex_table(t8, T / "tab8_strike.tex", index_name="Strategy")
    be = next((x for x in np.arange(0, 10.01, 0.25)
               if sharpe(((p["iv"] + x / 100) ** 2 - p["rv_next"]) / (2 * (p["iv"] + x / 100))) > 0), np.nan)
    macros["BreakEvenStrike"] = f"{be:.2f}"

    # ---- Figures
    fig_premium(p, F)
    fig_cumulative(scaled, F)
    fig_weights(w, F)

    # ---- Macros for the paper
    tex = "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in macros.items())
    (out / "numbers.tex").write_text(tex + "\n")
    print(f"[{code}] done -> {out}")
    print(t2.round(2).to_string())


def validate_calendar(code, daily, exp, T, macros):
    cal = check_calendar(daily, exp).set_index("offset_days")
    latex_table(cal.rename(columns={"median_abs_spread_change": "Median $|\\Delta \\ln(F_1/F_2)|$",
                                    "ratio_to_all_days": "Ratio to all days"}),
                T / "tabA1_calendar.tex", fmt="{:.4f}", index_name="Offset (trading days)")
    macros["RollSpikeRatio"] = f"{cal.loc[1, 'ratio_to_all_days']:.1f}"
    cal.to_csv(T / "tabA1_calendar.csv")
    peak = cal["ratio_to_all_days"].idxmax()
    if peak != 1:
        print(f"[{code}] WARNING: F1-F2 spread jumps peak at offset {peak:+d}, not +1. "
              "The contract calendar may not match the Bloomberg generic; "
              f"check the cycle in commodities.py or provide data/raw/{code}_calendar.csv")

def C_target_scale(r_ins, target=0.10):
    """Scale that gives `target` annual vol in the in-sample period."""
    return target / (r_ins.std() * np.sqrt(12))


if __name__ == "__main__":
    main()
