"""Tests that need no Bloomberg data.  Run with:  pytest"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from commodities import SPECS, contract_calendar, detect_code  # noqa: E402
from load_data import detect_commodity, load_daily  # noqa: E402
from make_synthetic_data import simulate, write_bloomberg_layout  # noqa: E402
from panel import build_panel  # noqa: E402
from rolls import check_calendar, front_returns  # noqa: E402
from strategies import rule_weights  # noqa: E402

DAYS = pd.bdate_range("2019-10-01", "2021-03-31")


def _cal(code):
    return contract_calendar(code, DAYS)


@pytest.mark.parametrize("code, delivery, last_trade, roll_ref", [
    # WTI May-2020: 25 Apr is a Saturday -> 4 business days before
    ("CL", "2020-05", "2020-04-21", "2020-04-21"),
    # WTI Apr-2020: 25 Mar is a business day -> 3 business days before
    ("CL", "2020-04", "2020-03-20", "2020-03-20"),
    # Natural gas Jan-2021: 3 business days before 1 Jan
    ("NG", "2021-01", "2020-12-29", "2020-12-29"),
    # Gold Dec-2020: third-last business day; first notice 30 Nov
    ("GC", "2020-12", "2020-12-29", "2020-11-30"),
    # Corn Dec-2020: business day before the 15th; first notice 30 Nov
    ("C", "2020-12", "2020-12-14", "2020-11-30"),
])
def test_contract_calendar_rules(code, delivery, last_trade, roll_ref):
    row = _cal(code).loc[pd.Period(delivery, "M")]
    assert row["last_trade"] == pd.Timestamp(last_trade)
    assert row["roll_ref"] == pd.Timestamp(roll_ref)


def test_contract_cycles():
    assert len(_cal("CL").loc["2020"]) == 12                         # every month
    assert list(_cal("C").loc["2020"].index.month) == [3, 5, 7, 9, 12]


@pytest.mark.parametrize("ticker, code", [("CL1 COMB Comdty", "CL"), ("C 1 Comdty", "C"),
                                          ("GC1 Comdty", "GC"), ("hg1 comdty", "HG")])
def test_detect_code(ticker, code):
    assert detect_code(ticker) == code


@pytest.fixture(scope="module", params=["CL", "GC", "C"])
def synthetic(request, tmp_path_factory):
    code = request.param
    path = tmp_path_factory.mktemp("data") / f"synthetic_{code}.xlsx"
    write_bloomberg_layout(simulate(code, start="2010-01-04", end="2014-12-31", seed=1), path, code)
    daily = load_daily(path)
    cal = contract_calendar(code, daily.index)
    front = front_returns(daily, cal)
    return code, path, daily, cal, front, build_panel(daily, front)


def test_loader_and_detection(synthetic):
    code, path, daily, *_ = synthetic
    assert detect_commodity(path) == code
    assert {"f1", "f2", "iv30", "iv1m", "iv2m"} <= set(daily.columns)
    assert daily.index.is_monotonic_increasing


def test_calendar_matches_generic_rolls(synthetic):
    _, _, daily, cal, *_ = synthetic
    table = check_calendar(daily, cal["last_trade"]).set_index("offset_days")
    assert table["ratio_to_all_days"].idxmax() == 1


def test_returns_ignore_roll_jumps(synthetic):
    """The generic F1 jumps when it rolls; the rolled position must not."""
    *_, front, _ = synthetic
    raw = np.log(synthetic[2]["f1"]).diff().reindex(front.index).abs()
    on_roll = front["generic_roll"]
    assert front["ret"].abs()[on_roll].median() < 0.5 * raw[on_roll].median()


def test_panel_is_monthly_and_complete(synthetic):
    p = synthetic[-1]
    assert p.index.is_unique
    assert p[["iv", "rv_next", "ret_short_var"]].notna().all().all()


def test_rules_use_no_future_information(synthetic):
    """Changing next month's realised variance must not change any weight."""
    p = synthetic[-1]
    w0 = rule_weights(p)
    p2 = p.copy()
    p2["rv_next"] = p2["rv_next"] * np.random.default_rng(0).uniform(0.1, 10, len(p2))
    pd.testing.assert_frame_equal(w0, rule_weights(p2))


def test_every_spec_has_valid_rules():
    for code in SPECS:
        assert len(contract_calendar(code, DAYS)) > 0, code


def test_calendar_file_french_excel_csv(tmp_path):
    """Semicolon-separated CSV with dd/mm/yyyy dates, as French Excel saves it."""
    from commodities import calendar_from_file
    f = tmp_path / "GC_calendar.csv"
    f.write_text("Ticker;LAST_TRADEABLE_DT;FUT_NOTICE_FIRST\n"
                 "GCZ0 Comdty;29/12/2020;30/11/2020\n"
                 "GCG1 Comdty;24/02/2021;29/01/2021\n", encoding="utf-8")
    cal = calendar_from_file(f, DAYS)
    assert list(cal["last_trade"]) == [pd.Timestamp("2020-12-29"), pd.Timestamp("2021-02-24")]
    assert list(cal["roll_ref"]) == [pd.Timestamp("2020-11-30"), pd.Timestamp("2021-01-29")]
