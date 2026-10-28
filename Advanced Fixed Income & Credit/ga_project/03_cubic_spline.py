"""
Question 2.1 -- Cubic spline of zero-coupon yields, knots at 2,5,10,15,20y.

Methodology choices (documented per the assignment's instructions):
  - We spline the ZERO-COUPON YIELD (BEY) directly, not the discount factor
    or the log-discount-factor, because Q2.1 explicitly asks to plot the
    "discount function (i.e., the cubic splines based ZC yields)" against the
    Q1.1 zero-coupon yields -- i.e. yields are the quantity of interest here.
  - Knots: exactly the 5 maturities given in the instructions (2,5,10,15,20y).
    We do NOT add extra knots at the very short end (e.g. 1y) or beyond 20y,
    to stay within the letter of the assignment; this means the spline is
    extrapolated by continuing the first/last cubic polynomial outside [2,20]y.
  - Boundary condition: "natural" cubic spline (zero second derivative at the
    two end knots), explicitly chosen consistently with the course's natural
    spline. SciPy's default boundary condition is instead "not-a-knot".
  - We fit on the BOOTSTRAPPED yields from Q1.1 (not on Deutsche Bank's model
    yields), since Q2.1 asks for a spline on "the zero-coupon yields obtained
    in Q1.1".
"""
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

curve = pd.read_csv('outputs/q1_bootstrapped_curve.csv')

knot_years = [2, 5, 10, 15, 20]
knots = curve[curve['Years'].isin(knot_years)].sort_values('Years')
assert len(knots) == 5, f"Expected 5 knots, found {len(knots)}"

cs = CubicSpline(knots['Years'].values, knots['ZeroYield_BEY'].values, bc_type='natural')

# Evaluate the spline at every bootstrapped maturity (0.5y steps).
# Inside [2,20]y, interpolate between the five selected knots.
# Outside [2,20]y, continue the first/last cubic polynomial.
# This interpolation does not statistically estimate or remove pricing noise.
curve['SplineYield'] = cs(curve['Years'])
curve['InKnotRange'] = curve['Years'].between(2, 20)
curve.to_csv('outputs/q2_spline_vs_bootstrap.csv', index=False)

# --- Plot: spline vs raw bootstrapped yields ---
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(curve['Years'], curve['ZeroYield_BEY']*100, 'o', ms=3, color='#4D4D4D',
        label='Bootstrapped zero yields (Q1.1, raw)')
ax.plot(curve['Years'], curve['SplineYield']*100, '-', lw=1.8, color='#9E1B32',
        label='Cubic spline (knots: 2,5,10,15,20y)')
ax.scatter(knots['Years'], knots['ZeroYield_BEY']*100, marker='D', s=55,
           color='#9E1B32', zorder=5, label='Spline knots')
ax.axvspan(0, 2, color='grey', alpha=0.08)
ax.axvspan(20, 26, color='grey', alpha=0.08)
ax.set_xlabel('Maturity (years)')
ax.set_ylabel('Zero-coupon yield, BEY (%)')
ax.set_title('Q2.1 -- Cubic spline vs. bootstrapped zero-coupon yield curve')
ax.legend(frameon=False)
ax.text(1, ax.get_ylim()[0]+0.1, 'extrapolated', fontsize=8, color='grey', style='italic')
ax.text(21, ax.get_ylim()[0]+0.1, 'extrapolated', fontsize=8, color='grey', style='italic')
fig.tight_layout()
fig.savefig('outputs/q2_spline_vs_bootstrap.png', dpi=150)
print("Saved plot to outputs/q2_spline_vs_bootstrap.png")

# residuals (fit error) within the knot range, for the write-up
within = curve[curve['InKnotRange']]
resid_bps = (within['SplineYield'] - within['ZeroYield_BEY']) * 10000
print(f"\nResidual (spline - bootstrap), within [2,20]y, in bps:")
print(f"  mean={resid_bps.mean():.2f}  std={resid_bps.std():.2f}  "
      f"min={resid_bps.min():.2f}  max={resid_bps.max():.2f}")
print(f"  largest absolute residuals at:")
print(within.assign(resid_bps=resid_bps).reindex(resid_bps.abs().sort_values(ascending=False).index)
      [['Years','ZeroYield_BEY','SplineYield','resid_bps']].head(8).to_string(index=False))
