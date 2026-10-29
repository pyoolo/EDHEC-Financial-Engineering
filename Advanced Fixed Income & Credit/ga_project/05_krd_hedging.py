"""
Part Filippo
Question 4.1 -- Shock the 10-year key rate by 50bps; impact on the Q2.1 spline
                 zero-coupon curve. Plot the difference.
Question 4.2 -- KRD of a $1M, equally-weighted (across all 52 bonds) portfolio
                 to that 10Y shock.
Question 4.3 -- KRD to each of the 5 key rates (2,5,10,15,20y) individually,
                 AND to each of the 52 semi-annual zero yields individually.
Extra       -- Small KRD hedge example (5 hedge bonds, 5 key rates).

Methodology & documented assumptions:
  - KEY-RATE SHOCK = BUMP THE SPLINE KNOT AND REFIT. The Q2.1 curve is a natural
    cubic spline through the zero yields at the knots 2,5,10,15,20y, so "shock
    the 10Y rate by 50bps" is implemented literally: add 50bps to the 10Y knot
    value, refit the natural spline, and take Delta-y(t) = bumped spline minus
    base spline at every maturity. Consequences (discussed in the output):
      * the shock is NOT local: a cubic spline ripples, with alternating signs,
        into neighbouring maturities;
      * the effect is amplified outside [2,20]y, where the spline is
        EXTRAPOLATED (the end cubic polynomial is continued, it is not flat);
      * the 5 key-rate shocks still sum EXACTLY to a parallel shift (spline
        interpolation is linear in the knot values) -- checked below.
  - For COMPARISON we also run the standard triangular ("tent") key-rate shock
    (Ho, 1992): +1x at its own knot, decaying linearly to 0 at the adjacent
    knots, held flat beyond the outer knots. It is local by construction.
  - PRICING CURVE: base prices use the Q1.1 BOOTSTRAPPED zero curve, which
    reprices all 52 bonds exactly, so the base NAV is exactly $1,000,000. The
    spline-implied Delta-y(t) is then added on top of the bootstrapped yields.
    (Pricing off the spline itself would give a base NAV that differs from
    $1M by the Q2.2 spline fitting residual, which would contaminate the KRD.)
  - "Equally invested in all bonds" (Q4.2) is interpreted as an equal DOLLAR
    MARKET VALUE in each of the 52 bonds: $1,000,000 / 52 ~= $19,230.77 each,
    translated into a face-value holding of (dollar amount / price) x 100.
  - KRD is reported three ways: (i) $ P&L of the +50bps shock (one-sided);
    (ii) KRD in YEARS, KRD_k = -(P_up - P_dn) / (2 * P * h) with h = 50bps
    (central difference, which removes most of the convexity error, so the
    KRDs add up to the effective duration); (iii) DV01 ($ per 1bp).
    NOTE the DV01 reported is the CENTRAL-difference one; the one-sided P&L
    implies a smaller $/bp because a 50bps move is large enough for convexity
    to matter (both are printed, so the text can quote them consistently).
  - Robustness: outputs/q4_2_robustness.csv recomputes the 10Y KRD under every
    defensible convention (spline vs tent shock, bootstrapped vs spline-priced
    base, cubic vs flat beyond 20y vs linear extrapolation, single-point bump,
    equal-face portfolio); outputs/q4_3_extrapolation_sensitivity.csv does the
    same for the five key-rate KRDs beyond the 20y knot.
  - Yields are Bond-Equivalent Yields (semi-annual compounding), as in Q1.1.
"""
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- style ----
# Figures are generated at the width they will occupy in the PDF
# (\textwidth = 16.4cm = 6.46in), so nothing is downscaled and the labels stay
# legible in print. dpi=300 gives ~300 dpi at final size.
RED, GREY, BAND = '#9E1B32', '#4D4D4D', '#B0B0B0'
SHOCK_COLORS = ['#7F7F7F', '#4C72B0', '#9E1B32', '#DD8452', '#55A868']
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8,
    'axes.titlesize': 8.5,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.color': '#DDDDDD',
    'grid.linewidth': 0.6,
    'axes.axisbelow': True,
    'axes.edgecolor': GREY,
    'axes.linewidth': 0.7,
    'xtick.color': GREY,
    'ytick.color': GREY,
})

BASE = Path(__file__).resolve().parent
OUT = BASE / 'outputs'
SETTLEMENT = date(2003, 8, 15)

# ------------------------------------------------------------------ data ----
bonds = pd.read_csv(BASE / 'data/exhibit1_bonds.csv')
bonds = bonds[pd.to_numeric(bonds['Coupon'], errors='coerce').notna()].copy()
bonds['Maturity'] = pd.to_datetime(bonds['Maturity'])
bonds['Coupon'] = bonds['Coupon'].astype(float)
bonds['Price'] = bonds['Price'].astype(float)
bonds = bonds.sort_values('Maturity').reset_index(drop=True)
bonds['Periods'] = ((bonds['Maturity'].dt.year - SETTLEMENT.year) * 2
                    + (bonds['Maturity'].dt.month - SETTLEMENT.month) // 6)
n_bonds = len(bonds)
assert (bonds['Periods'].values == np.arange(1, n_bonds + 1)).all(), "Expected one bond per semi-annual node"

curve = pd.read_csv(OUT / 'q2_spline_vs_bootstrap.csv')
t = curve['Years'].values                    # 0.5, 1.0, ..., 26.0
y_boot = curve['ZeroYield_BEY'].values       # Q1.1 bootstrapped zero yields (pricing base)
y_spline = curve['SplineYield'].values       # Q2.1 spline

KNOTS = [2, 5, 10, 15, 20]
SHOCK = 0.0050                               # 50 bps
knot_y = curve.set_index('Years').loc[KNOTS, 'ZeroYield_BEY'].values

# Cash-flow matrix: CF[i, k] = cash flow of bond i at period k+1, per 100 face
CF = np.zeros((n_bonds, n_bonds))
for i, row in bonds.iterrows():
    CF[i, :row['Periods']] = row['Coupon'] / 2.0
    CF[i, row['Periods'] - 1] += 100.0

def disc(y):
    """BEY (semi-annual compounding) zero yields -> discount factors."""
    return (1.0 + y / 2.0) ** (-2.0 * t)

# Portfolio: $1M, equal dollar market value in each bond
NAV_TARGET = 1_000_000.0
face = (NAV_TARGET / n_bonds) / bonds['Price'].values * 100.0   # face held per bond
port_cf = (face / 100.0) @ CF                                   # portfolio cash flow per period

# Alternative reading of "equally invested": equal FACE in each bond ($1M/52 of
# face, i.e. a market value of $1.19M). Kept only for the robustness table: it
# weights each bond by its price, overweighting the high-coupon 10.5-20y bonds
# (prices 130-174) relative to the near-par short bonds (~103).
face_eq = np.full(n_bonds, NAV_TARGET / n_bonds)
port_cf_eq = (face_eq / 100.0) @ CF

def nav(y):
    return port_cf @ disc(y)

# Checks on the pricing base
assert np.allclose(CF @ disc(y_boot), bonds['Price'].values, atol=1e-6), "Bootstrap does not reprice bonds"
P0 = nav(y_boot)
assert abs(P0 - NAV_TARGET) < 1e-4
print(f"Base NAV (priced off the Q1.1 bootstrapped curve): ${P0:,.2f}  (= target $1,000,000)")
print(f"[for reference: pricing off the Q2.1 spline would give ${nav(y_spline):,.2f}, "
      f"i.e. the Q2.2 fit residual -- not used]\n")

# --------------------------------------------------------------- shocks -----
def spline_shock(j, size=SHOCK):
    """Delta-y(t): bump knot j by `size`, refit the natural cubic spline, difference vs base spline."""
    bumped = knot_y.copy()
    bumped[j] += size
    return CubicSpline(KNOTS, bumped, bc_type='natural')(t) - CubicSpline(KNOTS, knot_y, bc_type='natural')(t)

def tent_shock(j, size=SHOCK):
    """Ho (1992) triangular shock, flat beyond the outer knots."""
    k = KNOTS[j]
    shock = np.zeros_like(t)
    if j == 0:
        shock[t <= k] = 1.0
    else:
        kp = KNOTS[j - 1]
        m = (t > kp) & (t <= k)
        shock[m] = (t[m] - kp) / (k - kp)
    if j == len(KNOTS) - 1:
        shock[t > k] = 1.0
    else:
        kn = KNOTS[j + 1]
        m = (t > k) & (t < kn)
        shock[m] = (kn - t[m]) / (kn - k)
    return size * shock

spl = [spline_shock(j) for j in range(5)]
tnt = [tent_shock(j) for j in range(5)]
assert np.allclose(sum(spl), SHOCK), "Spline key-rate shocks do not sum to a parallel shift"
assert np.allclose(sum(tnt), SHOCK), "Tent key-rate shocks do not sum to a parallel shift"
print("Checks passed: both families of 5 key-rate shocks sum to a +50bps parallel shift at every maturity.\n")

def krd(dy, h=SHOCK):
    """KRD of the portfolio for a shock profile dy that has size h at its key rate."""
    up = nav(y_boot + dy)
    dn = nav(y_boot - dy)
    return {'PnL_up': up - P0,
            'KRD_years': -(up - dn) / (2 * P0 * h),
            'DV01': -(up - dn) / (2 * h) * 1e-4}

def krd_from(base_y, dy, h=SHOCK, pcf_local=None):
    """Same KRD, but around an arbitrary base curve and/or portfolio cash flow.
    Used only by the robustness table: it lets us vary the pricing base
    (bootstrapped vs spline) and the portfolio definition (equal $MV vs face)."""
    cf = port_cf if pcf_local is None else pcf_local
    P = cf @ disc(base_y)
    up, dn = cf @ disc(base_y + dy), cf @ disc(base_y - dy)
    return {'P0': P, 'PnL_up': up - P,
            'KRD_years': -(up - dn) / (2 * P * h),
            'DV01': -(up - dn) / (2 * h) * 1e-4}

def flat_tail_shock(j, size=SHOCK):
    """Bump knot j and re-fit. Beyond 20Y the outermost knot owns the tail
    (shock held at +size); the other four are held flat (zero). Below 2Y the
    end cubic is kept, as in Q4.1. The five shocks still sum to a parallel shift."""
    dy = spline_shock(j, size).copy()
    dy[t > KNOTS[-1]] = size if j == len(KNOTS) - 1 else 0.0
    return dy

def linear_ext(j):
    """Extend the spline shock beyond the outer knots along the boundary slope."""
    bumped = knot_y.copy(); bumped[j] += SHOCK
    s_up = CubicSpline(KNOTS, bumped, bc_type='natural')
    s_b = CubicSpline(KNOTS, knot_y, bc_type='natural')
    out = np.zeros_like(t)
    inside = (t >= KNOTS[0]) & (t <= KNOTS[-1])
    out[inside] = s_up(t[inside]) - s_b(t[inside])
    lo, hi = t < KNOTS[0], t > KNOTS[-1]
    out[lo] = (s_up(KNOTS[0], 1) - s_b(KNOTS[0], 1)) * (t[lo] - KNOTS[0])
    out[hi] = (s_up(KNOTS[-1], 1) - s_b(KNOTS[-1], 1)) * (t[hi] - KNOTS[-1])
    return out

def ix(years):
    return int(np.where(t == years)[0][0])

# ================= Q4.1: shock the 10Y key rate, impact on the Q2.1 curve =================
j10 = KNOTS.index(10)
d10 = spl[j10]
fig, axes = plt.subplots(1, 2, figsize=(6.45, 2.75), layout='constrained')
for a in axes:
    a.axvspan(0, 2, color=BAND, alpha=0.18, lw=0, zorder=0)
    a.axvspan(20, 26, color=BAND, alpha=0.18, lw=0, zorder=0)
    a.axhline(0, color='black', lw=0.6, zorder=1)
    a.set_xlim(0, 26.5)
    a.set_xlabel('Maturity (years)')

# (a) the 10Y shock: spline re-fit vs the local tent
ax = axes[0]
ax.plot(t, d10 * 1e4, color=RED, lw=1.6, label='Spline re-fit (Q2.1 model)', zorder=3)
ax.plot(t, tnt[j10] * 1e4, color=GREY, lw=1.1, ls='--', label='Triangular tent (comparison)', zorder=2)
ax.plot([k for k in KNOTS if k != 10], np.zeros(len(KNOTS) - 1), 'o', ms=3.4, mfc='white',
        mec=GREY, mew=0.8, ls='none', label='Knots left unchanged', zorder=4)
ax.text(10.4, 51.5, '+50bps', fontsize=7, color=RED, ha='left', va='center')
ax.text(25.6, -15.5, '-11.5bps at 26y', fontsize=7, color=RED, ha='right', va='center')
ax.set_ylabel('Yield change (bps)')
ax.set_title('(a) 10Y shock: spline re-fit vs tent')
ax.set_ylim(-22, 57)
ax.legend(frameon=False, loc='lower left', handlelength=1.5, borderaxespad=0.2)

# (b) all five key-rate shocks
ax = axes[1]
for j, k in enumerate(KNOTS):
    ax.plot(t, spl[j] * 1e4, color=SHOCK_COLORS[j], lw=1.7 if k == 10 else 1.0,
            label=f'{k}Y', zorder=3 if k == 10 else 2)
    ax.axvline(k, color='#999999', ls=':', lw=0.6, zorder=1)
ax.set_ylabel('Yield change (bps)')
ax.set_title('(b) The five key-rate shocks')
ax.set_ylim(-62, 120)
ax.legend(frameon=False, ncol=5, loc='upper center', handlelength=1.2,
          columnspacing=1.0, handletextpad=0.4, borderaxespad=0.2)
fig.savefig(OUT / 'q4_1_shock_10y.png', dpi=300)

lo, hi = d10[t < 5] * 1e4, d10[t > 20] * 1e4
print("Q4.1 -- impact of +50bps on the 10Y knot (spline refit):")
print(f"  Peak change {d10.max()*1e4:.1f}bps at {t[d10.argmax()]:.1f}y; change at 10y = {d10[ix(10)]*1e4:.1f}bps; "
      f"at the other knots 2/5/15/20y: " + ", ".join(f"{d10[ix(k)]*1e4:.1f}" for k in (2, 5, 15, 20)) + " bps (zero by construction).")
print(f"  NOT local: between knots the cubic ripples, e.g. {d10[ix(7)]*1e4:+.1f}bps at 7y, {d10[ix(12)]*1e4:+.1f}bps at 12y, "
      f"{d10[ix(17)]*1e4:+.1f}bps at 17y; below 5y the change ranges {lo.min():+.1f}..{lo.max():+.1f}bps.")
print(f"  Extrapolated region >20y (end cubic is continued, not flat): {hi.min():+.1f}..{hi.max():+.1f}bps "
      f"(26y: {d10[ix(26)]*1e4:+.1f}bps).")
print(f"  Tent comparison: tent gives exactly 0 beyond 5..20y and a linear ramp to +50bps at 10y.")
print("Saved outputs/q4_1_shock_10y.png\n")

# ================= Q4.2: portfolio KRD to the 10Y shock =================
r = krd(d10)
print("Q4.2 -- approach: equal-$ portfolio of the 52 bonds, repriced off the bootstrapped curve + spline Delta-y(t).")
print(f"  Base NAV ${P0:,.2f}; shocked NAV ${P0 + r['PnL_up']:,.2f}")
print(f"  KRD_10Y: P&L = ${r['PnL_up']:,.2f} ({r['PnL_up']/P0*1e4:.1f} bps of NAV) for +50bps; "
      f"KRD = {r['KRD_years']:.3f} years; DV01 = ${r['DV01']:,.2f} per bp")
dv01_one_sided = -r['PnL_up'] / SHOCK * 1e-4
print(f"  DV01 basis: central ${r['DV01']:,.2f}/bp vs one-sided ${dv01_one_sided:,.2f}/bp "
      f"(convexity gap ${r['DV01']-dv01_one_sided:,.2f}); P&L(+50bps) != -50 x central DV01")
rt = krd(tent_shock(j10))
print(f"  [tent shock instead: P&L = ${rt['PnL_up']:,.2f}, KRD = {rt['KRD_years']:.3f}y]\n")

# ================= Q4.2 robustness: what actually moves the number =================
dy_10_only = np.zeros_like(t)
dy_10_only[ix(10)] = SHOCK
variants = [
    ('Spline refit, bootstrapped base (REPORTED)', krd_from(y_boot, d10)),
    ('Spline refit, spline-priced base', krd_from(y_spline, d10)),
    ('Tent, bootstrapped base', krd_from(y_boot, tnt[j10])),
    ('Tent, spline-priced base', krd_from(y_spline, tnt[j10])),
    ('Spline refit, tail held flat beyond 20Y', krd_from(y_boot, flat_tail_shock(j10))),
    ('Spline refit, linear extension outside [2,20]y', krd_from(y_boot, linear_ext(j10))),
    ('Single-point 10Y bump (10Y zero yield only)', krd_from(y_boot, dy_10_only)),
    ('Equal-FACE portfolio, bootstrapped base', krd_from(y_boot, d10, pcf_local=port_cf_eq)),
]
rob = pd.DataFrame([{'Variant': v, 'Base_NAV_$': a['P0'], 'PnL_up_$': a['PnL_up'],
                     'KRD_years': a['KRD_years'], 'DV01_$': a['DV01']} for v, a in variants])
rob.to_csv(OUT / 'q4_2_robustness.csv', index=False)
print("Q4.2 robustness -- the KRD is robust to the pricing engine, not to the shock design:")
print(rob.round(3).to_string(index=False))
print(f"  Range of knot-based conventions: {rob['KRD_years'].iloc[:6].min():.2f}-"
      f"{rob['KRD_years'].iloc[:6].max():.2f}y; single-point bump is the zero-yield KRD of Q4.3(a), not the knot KRD.\n")

# ================= Q4.3 (a): the 5 key rates =================
rows = []
for j, k in enumerate(KNOTS):
    a, b = krd(spl[j]), krd(tnt[j])
    rows.append({'KeyRate': k,
                 'Spline_PnL_$': a['PnL_up'], 'Spline_PnL_pctNAV': a['PnL_up'] / P0 * 100,
                 'Spline_KRD_years': a['KRD_years'], 'Spline_DV01_$': a['DV01'],
                 'Tent_PnL_$': b['PnL_up'], 'Tent_KRD_years': b['KRD_years'], 'Tent_DV01_$': b['DV01']})
krd_df = pd.DataFrame(rows)
krd_df.to_csv(OUT / 'q4_3_krd_by_keyrate.csv', index=False)
print("Q4.3(a) -- KRD to each key rate (+50bps at each knot):")
print(krd_df.round(3).to_string(index=False))

eff_dur = -(nav(y_boot + SHOCK) - nav(y_boot - SHOCK)) / (2 * P0 * SHOCK)
parallel_pnl = nav(y_boot + SHOCK) - P0
print(f"\n  Sum of spline KRDs (years) = {krd_df['Spline_KRD_years'].sum():.4f}; "
      f"sum of tent KRDs = {krd_df['Tent_KRD_years'].sum():.4f}; parallel effective duration = {eff_dur:.4f}")
print(f"  Sum of one-sided key-rate P&Ls = ${krd_df['Spline_PnL_$'].sum():,.2f} vs. parallel +50bps P&L = ${parallel_pnl:,.2f} "
      f"(difference ${krd_df['Spline_PnL_$'].sum()-parallel_pnl:,.2f} = convexity/cross terms; central-difference KRDs remove most of it).")
# tolerance: spline shocks reach ~100bps in the extrapolated zone, so a small non-linearity remains
assert abs(krd_df['Spline_KRD_years'].sum() - eff_dur) < 0.05

# --- where the 15Y/20Y split comes from: sensitivity to the >20y extension -----
assert np.allclose(sum(flat_tail_shock(j) for j in range(5)), SHOCK), "Flat-tail shocks do not sum to a parallel shift"
sens = pd.DataFrame([{'KeyRate': k,
                      'EndCubic_KRD': krd(spl[j])['KRD_years'],
                      'FlatTail_KRD': krd(flat_tail_shock(j))['KRD_years']}
                     for j, k in enumerate(KNOTS)])
sens.to_csv(OUT / 'q4_3_extrapolation_sensitivity.csv', index=False)
print("\nQ4.3 -- sensitivity of the key-rate profile to the >20y extension:")
print(sens.round(3).to_string(index=False))
print(f"  sums: end cubic {sens['EndCubic_KRD'].sum():.4f}y vs flat tail {sens['FlatTail_KRD'].sum():.4f}y "
      f"(effective duration {eff_dur:.4f}y).")
print("  Both conventions sum to a +50bps parallel shift at every maturity, so both give a complete")
print("  decomposition; they differ only in how the extrapolated >20y tail is allocated between")
print("  the 15Y and 20Y buckets (0.81/3.06y under the end cubic vs 1.85/2.09y under the flat tail).\n")

x = np.arange(5); w = 0.38
fig2, ax2 = plt.subplots(figsize=(4.65, 3.0), layout='constrained')
ax2.bar(x - w/2, krd_df['Spline_KRD_years'], w, color=RED, label='Spline re-fit (Q2.1)')
ax2.bar(x + w/2, krd_df['Tent_KRD_years'], w, color=GREY, label='Triangular tent')
ax2.set_xticks(x); ax2.set_xticklabels([f'{k}Y' for k in KNOTS])
ax2.set_ylabel('Key-rate duration (years)')
ax2.set_ylim(0, krd_df[['Spline_KRD_years', 'Tent_KRD_years']].to_numpy().max() * 1.20)
ax2.set_title('Key-rate duration by key rate')
ax2.legend(frameon=False, loc='upper left', handlelength=1.2, borderaxespad=0.3)
for xi, v in zip(x - w/2, krd_df['Spline_KRD_years']):
    ax2.text(xi, v + 0.03, f'{v:.2f}', ha='center', va='bottom', fontsize=6.5, color=RED)
for xi, v in zip(x + w/2, krd_df['Tent_KRD_years']):
    ax2.text(xi, v + 0.03, f'{v:.2f}', ha='center', va='bottom', fontsize=6.5, color=GREY)
fig2.savefig(OUT / 'q4_3_krd_profile.png', dpi=300)
print("Saved outputs/q4_3_krd_profile.png\n")

# ================= Q4.3 (b): each of the 52 zero yields individually =================
krd52 = []
for i in range(n_bonds):
    dy = np.zeros(n_bonds); dy[i] = SHOCK
    a = krd(dy)
    krd52.append({'Years': t[i], 'PnL_$': a['PnL_up'], 'KRD_years': a['KRD_years'], 'DV01_$': a['DV01']})
krd52 = pd.DataFrame(krd52)
krd52.to_csv(OUT / 'q4_3_krd_by_maturity.csv', index=False)
assert abs(krd52['KRD_years'].sum() - eff_dur) < 0.01
print(f"Q4.3(b) -- sum of the 52 zero-yield KRDs = {krd52['KRD_years'].sum():.4f}y (= parallel effective duration {eff_dur:.4f}y)")
print(f"  largest single-maturity KRD: {krd52['KRD_years'].max():.3f}y at {krd52.loc[krd52['KRD_years'].idxmax(),'Years']:.1f}y")

fig3, ax3 = plt.subplots(figsize=(5.17, 2.8), layout='constrained')
ax3.bar(krd52['Years'], krd52['KRD_years'], width=0.42, color=RED, zorder=2)
for i, k in enumerate(KNOTS):
    ax3.axvline(k, color='#999999', ls=':', lw=0.7, zorder=1,
                label='Key-rate knots' if i == 0 else None)
ax3.set_xticks([0, 2, 5, 10, 15, 20, 25])
ax3.set_xlim(0, 26.5)
ax3.set_ylim(0, krd52['KRD_years'].max() * 1.14)
ax3.set_xlabel('Zero-coupon maturity (years)'); ax3.set_ylabel('KRD (years)')
ax3.set_title('KRD by zero-coupon maturity')
ax3.legend(frameon=False, loc='upper right', handlelength=1.2, borderaxespad=0.3)
ax3.text(krd52.loc[krd52['KRD_years'].idxmax(), 'Years'],
         krd52['KRD_years'].max(), f"{krd52['KRD_years'].max():.3f}",
         fontsize=6.5, color=GREY, ha='center', va='bottom',
         bbox=dict(facecolor='white', edgecolor='none', pad=0.6))
fig3.savefig(OUT / 'q4_3_krd_by_maturity.png', dpi=300)
print("Saved outputs/q4_3_krd_by_maturity.png\n")

# ================= Extra: KRD hedge with 5 bonds =================
# Hedge instruments: the bonds maturing at the key-rate dates (periods 4, 10, 20, 30, 40).
hedge_idx = [int(2 * k) - 1 for k in KNOTS]
def bond_dv01_vs_keyrate(i, j):
    """$ DV01 (per 100 face, per 1bp) of bond i to key rate j (central difference on the spline shock)."""
    d = spl[j]
    return -(CF[i] @ (disc(y_boot + d) - disc(y_boot - d))) / (2 * SHOCK) * 1e-4
A = np.array([[bond_dv01_vs_keyrate(i, j) for i in hedge_idx] for j in range(5)])   # rows: key rate, cols: hedge bond
target = np.array([krd(spl[j])['DV01'] for j in range(5)])
hedge_face = -np.linalg.solve(A, target) * 100.0         # face to hold in each hedge bond (negative = short)
hedge_cf = (hedge_face / 100.0) @ CF[hedge_idx]
print("Hedge example -- face amounts of the bonds maturing at 2/5/10/15/20y that zero the 5 key-rate DV01s:")
for k, i, f in zip(KNOTS, hedge_idx, hedge_face):
    print(f"  {k:>2}Y ({bonds.loc[i,'Maturity'].date()}, {bonds.loc[i,'Coupon']:>5}%): face ${f:>10,.0f}")
print(f"  Hedge market value ${(hedge_face/100*bonds.loc[hedge_idx,'Price'].values).sum():,.0f} "
      "(funding leg not modelled). Residual P&L (unhedged -> hedged):")
res = []
for j, k in enumerate(KNOTS):
    dy = spl[j]
    unh = nav(y_boot + dy) - P0
    res.append((f"{k}Y", unh, unh + hedge_cf @ (disc(y_boot + dy) - disc(y_boot))))
unh = nav(y_boot + SHOCK) - P0
res.append(("Parallel", unh, unh + hedge_cf @ (disc(y_boot + SHOCK) - disc(y_boot))))
print("  " + " | ".join(f"{k}: ${u:,.0f} -> ${h:,.0f}" for k, u, h in res))

# Flat-tail hedge comparison (model-risk sensitivity of the hedge legs)
fts = [flat_tail_shock(j) for j in range(5)]
A_ft = np.array([[-(CF[i] @ (disc(y_boot + fts[j]) - disc(y_boot - fts[j]))) / (2 * SHOCK) * 1e-4
                  for i in hedge_idx] for j in range(5)])
target_ft = np.array([krd(fts[j])['DV01'] for j in range(5)])
hedge_face_ft = -np.linalg.solve(A_ft, target_ft) * 100.0
print("\n  [Flat-tail convention instead: face amounts "
      + ", ".join(f"{k}Y ${f:,.0f}" for k, f in zip(KNOTS, hedge_face_ft)) + "]")
