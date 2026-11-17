// Builds the presentation:  cd presentation && npm install && npm run build
// Uses the chart images in ./img (exported from the paper figures by run_all.sh).
const pptxgen = require("pptxgenjs");
const path = require("path");
let applyTheme = null;
try { applyTheme = require(process.env.APPLY_THEME || "./apply_theme.js").applyTheme; } catch (e) { /* optional: writes theme colours */ }

// ------------------------------------------------------------------ identity
// "Research note" style: sans-serif only, left-aligned titles with a numbered kicker,
// no cards, hairline dividers, burnt-orange signal colour on ink black / white.
const THEME = {
  name: "Research Note",
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  colors: {
    dk1: "141414", lt1: "FFFFFF", dk2: "141414", lt2: "F3F3F3",
    accent1: "D9480F",  // burnt orange: the signal colour
    accent2: "1F4E79",  // expectations blue (matches charts)
    accent3: "C0392B",  // oil red (matches charts)
    accent4: "8C8C8C",  // grey text
    accent5: "D0D0D0",  // hairlines
    accent6: "2B7A4B",  // supported
    hlink: "1F4E79", folHlink: "8C8C8C",
  },
};
const IMG = (f) => path.join(__dirname, "img", f + ".png");
const MONO = "Courier New";
const TOTAL = 12;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Oil Shocks, Inflation Pass-Through and the Pricing of Inflation Risk";
pres.author = "Group project";
// Theme colours: if apply_theme.js is available the deck uses real theme (scheme) colours,
// otherwise the same colours are written as plain hex so the deck looks identical.
const H = THEME.colors;
const C = applyTheme ? pres.SchemeColor : { text1: H.dk1, text2: H.dk2, background1: H.lt1, background2: H.lt2,
  accent1: H.accent1, accent2: H.accent2, accent3: H.accent3, accent4: H.accent4, accent5: H.accent5, accent6: H.accent6 };

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "COVER",
  background: { color: C.accent1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.6, w: 10.5, h: 3.0, fontSize: 46, bold: true, color: C.text1, valign: "bottom", align: "left", fontFace: "Arial" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 4.9, w: 10.5, h: 1.6, fontSize: 18, color: C.text1, valign: "top", align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "PAGE",
  background: { color: C.background1 },
  margin: [0.5, 0.7, 0.6, 0.7],
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: 0.7, y: 0.42, w: 9, h: 0.3, fontSize: 12, bold: true, color: C.accent1, charSpacing: 2, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 0.72, w: 11.9, h: 0.75, fontSize: 28, bold: true, color: C.text1, align: "left", valign: "top", margin: 0, fontFace: "Arial" }, text: "" } },
    { text: { text: "OIL SHOCKS & INFLATION EXPECTATIONS", options: { x: 0.7, y: 7.02, w: 6, h: 0.25, fontSize: 9, color: C.accent4, charSpacing: 2, isTextBox: true, margin: 0 } } },
  ],
});
pres.defineSlideMaster({
  title: "CLOSING",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: 0.7, y: 0.42, w: 9, h: 0.3, fontSize: 12, bold: true, color: C.accent1, charSpacing: 2, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 0.72, w: 11.9, h: 0.75, fontSize: 28, bold: true, color: C.background1, align: "left", valign: "top", margin: 0, fontFace: "Arial" }, text: "" } },
  ],
});

// ------------------------------------------------------------------ helpers
let pageNo = 0;
function page(kicker, title, section, master = "PAGE") {
  pageNo += 1;
  const sl = pres.addSlide({ masterName: master, sectionTitle: section });
  sl.addText(kicker.toUpperCase(), { placeholder: "kicker" });
  sl.addText(title, { placeholder: "title" });
  sl.addText(`${String(pageNo).padStart(2, "0")} / ${TOTAL}`, { x: 11.6, y: 7.02, w: 1.0, h: 0.25, fontSize: 9, color: master === "CLOSING" ? C.accent5 : C.accent4, align: "right", margin: 0, isTextBox: true, objectName: "page number" });
  return sl;
}
const T = (sl, text, o, name) => sl.addText(text, Object.assign({ margin: 0, isTextBox: true, color: C.text1, fontSize: 14, valign: "top", objectName: name }, o));
function hline(sl, x, y, w, name, color) { sl.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color: color || C.accent5, width: 0.75 }, objectName: name }); }
function vline(sl, x, y, h, name, color) { sl.addShape(pres.shapes.LINE, { x, y, w: 0, h, line: { color: color || C.accent5, width: 0.75 }, objectName: name }); }
function stats(sl, items, y, x0 = 0.7, wTot = 11.9, h = 1.15, dark = false) {
  const w = wTot / items.length;
  items.forEach(([big, label, col], i) => {
    const x = x0 + i * w;
    if (i > 0) vline(sl, x, y + 0.05, h - 0.1, "stat divider " + i, dark ? "3A3A3A" : null);
    T(sl, big, { x: x + (i ? 0.25 : 0), y, w: w - 0.35, h: 0.62, fontSize: 32, bold: true, color: col || C.accent1, valign: "bottom" }, "stat value " + (i + 1));
    T(sl, label, { x: x + (i ? 0.25 : 0), y: y + 0.66, w: w - 0.4, h: h - 0.66, fontSize: 12, color: dark ? C.accent5 : C.accent4 }, "stat label " + (i + 1));
  });
}
function verdict(sl, x, y, text, col, name) {
  sl.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.16, h: 0.16, fill: { color: col }, line: { color: col }, objectName: name + " marker" });
  T(sl, text, { x: x + 0.26, y: y - 0.07, w: 3, h: 0.3, fontSize: 12, bold: true, color: col, charSpacing: 1 }, name + " verdict");
}
function num(sl, x, y, n, name, col) {
  sl.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.36, h: 0.36, fill: { color: col || C.accent1 }, line: { color: col || C.accent1 }, objectName: name + " number box" });
  T(sl, String(n), { x, y, w: 0.36, h: 0.36, fontSize: 13, bold: true, color: C.background1, align: "center", valign: "middle" }, name + " number");
}
function img(sl, file, x, y, w, ratio, name) { sl.addImage({ path: IMG(file), x, y, w, h: w / ratio, objectName: name, altText: name }); return y + w / ratio; }
const note = (sl, text, y) => T(sl, text, { x: 0.7, y, w: 11.9, h: 0.28, fontSize: 10, color: C.accent4, italic: true }, "chart note");

// ================================================================== 1 COVER
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "COVER", sectionTitle: "Introduction" }); pageNo += 1;
s.addText("Oil shocks, inflation pass-through and the pricing of inflation risk", { placeholder: "title" });
s.addText([
  { text: "Evidence from the United States and the euro area, 1990–2026", options: { breakLine: true } },
  { text: "MACRO & CAPITAL MARKETS · GROUP PROJECT · NOVEMBER 2026", options: { fontSize: 12, bold: true, charSpacing: 2, color: C.background1 } },
], { placeholder: "body" });
s.addNotes("(30 sec) Introduce the group and the topic: how oil shocks feed into inflation, and whether markets price that correctly.");

// ================================================================== 2 QUESTION
s = page("01 — Question", "When oil moves, do inflation and markets follow?", "Introduction");
T(s, "Mechanism: cash flows vs discount rates", { x: 0.7, y: 1.75, w: 5.6, h: 0.35, fontSize: 15, bold: true }, "mechanism header");
const flow = [
  ["Oil shock", "Higher costs for firms, higher prices for households"],
  ["Temporary or lasting?", "Headline only, or second-round effects into core and wages"],
  ["Expectations and rates", "If long-term expectations rise, central banks tighten and assets reprice"],
];
flow.forEach(([h, d], i) => {
  const y = 2.35 + i * 1.4;
  T(s, String(i + 1).padStart(2, "0"), { x: 0.7, y, w: 0.9, h: 0.6, fontSize: 30, bold: true, color: C.accent1 }, "step number " + (i + 1));
  T(s, h, { x: 1.65, y: y + 0.02, w: 4.6, h: 0.35, fontSize: 16, bold: true }, "step title " + (i + 1));
  T(s, d, { x: 1.65, y: y + 0.4, w: 4.6, h: 0.7, fontSize: 14, color: "4A4A4A" }, "step text " + (i + 1));
  if (i < 2) hline(s, 0.7, y + 1.2, 5.6, "step rule " + (i + 1));
});
vline(s, 6.75, 1.8, 4.9, "column divider");
T(s, "Hypotheses", { x: 7.2, y: 1.75, w: 5.4, h: 0.35, fontSize: 15, bold: true }, "hypotheses header");
[["H1", "Strong, fast pass-through to headline; weak and lagged to core"],
 ["H2", "Stronger pass-through in the euro area (energy importer)"],
 ["H3", "5-year expectations react; the 5y5y forward does not"],
 ["H4", "Pass-through has changed over time, larger after 2020"]].forEach(([k, t], i) => {
  const y = 2.35 + i * 1.08;
  T(s, k, { x: 7.2, y, w: 0.8, h: 0.5, fontSize: 20, bold: true, color: C.accent2 }, k + " label");
  T(s, t, { x: 8.05, y: y + 0.02, w: 4.55, h: 0.75, fontSize: 14 }, k + " text");
  if (i < 3) hline(s, 7.2, y + 0.88, 5.4, k + " rule");
});
s.addNotes("(2 min) Research question: when oil prices move, how much and how fast does it show up in realized inflation, and do market inflation expectations react consistently? Link to Lecture 1: an oil shock is first a cash-flow shock; it only becomes a discount-rate shock if investors think it is a lasting inflation problem. Then read the four hypotheses.");

// ================================================================== 3 DATA & METHOD
pres.addSection({ title: "Data and method" });
s = page("02 — Data and method", "Two datasets, four complementary tests", "Data and method");
[["MONTHLY · 1990–2026", "Brent, WTI, Bloomberg Commodity Index\nUS CPI and euro area HICP, headline and core", "→ pass-through to realized inflation"],
 ["DAILY · 2004–2026", "US 5y breakeven, US 5y5y swap\nEuro area 5y and 5y5y swaps, S&P 500", "→ market reaction, aggregated to weekly"]].forEach(([h, d, u], i) => {
  const y = 1.8 + i * 1.95;
  T(s, h, { x: 0.7, y, w: 5.6, h: 0.3, fontSize: 12, bold: true, color: C.accent1, charSpacing: 2 }, "data title " + (i + 1));
  T(s, d, { x: 0.7, y: y + 0.4, w: 5.6, h: 0.8, fontSize: 15 }, "data series " + (i + 1));
  T(s, u, { x: 0.7, y: y + 1.25, w: 5.6, h: 0.35, fontSize: 13, color: C.accent4, italic: true }, "data use " + (i + 1));
});
hline(s, 0.7, 3.6, 5.6, "data rule");
T(s, "Source: Bloomberg. Shock = monthly change in average Brent (USD, nominal; deflating changes results by < 0.03 pp).", { x: 0.7, y: 5.75, w: 5.6, h: 0.7, fontSize: 11, color: C.accent4 }, "data note");
vline(s, 6.75, 1.8, 4.9, "column divider");
[["Local projections", "Inflation response 0–24 months after the shock, Newey–West s.e."],
 ["Weekly market regressions", "Δ breakeven / swap on Brent return, S&P 500 control"],
 ["Sign + narrative restrictions", "Split oil moves into global demand vs oil-specific (supply)"],
 ["Out-of-sample test", "Model estimated to 2019 predicts inflation after the 2026 shock"]].forEach(([h, d], i) => {
  const y = 1.8 + i * 1.18;
  num(s, 7.2, y + 0.02, i + 1, "method " + (i + 1));
  T(s, h, { x: 7.8, y, w: 4.8, h: 0.38, fontSize: 16, bold: true }, "method title " + (i + 1));
  T(s, d, { x: 7.8, y: y + 0.4, w: 4.8, h: 0.6, fontSize: 14, color: "4A4A4A" }, "method text " + (i + 1));
});
s.addNotes("(1.5 min) Two datasets from Bloomberg: monthly 1990-2026 for inflation, daily 2004-2026 for markets, aggregated to weekly. We use the monthly average Brent price because CPI measures average prices over the month. Four tests: local projections, weekly market regressions, a supply-versus-demand split, and an out-of-sample test on the 2026 shock.");

// ================================================================== 4-5 CODE
function codeSlide(kicker, title, lines, anns, foot, notes) {
  const sl = page(kicker, title, "Data and method");
  sl.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 1.75, w: 7.55, h: 4.85, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "code panel" });
  T(sl, lines.map((l, i) => ({ text: l.t || " ", options: { color: l.c ? C.accent1 : C.text1, bold: !!l.c, breakLine: i < lines.length - 1 } })),
    { x: 0.9, y: 1.92, w: 7.2, h: 4.55, fontFace: MONO, fontSize: 12 }, "code");
  anns.forEach(([n, h, d], i) => {
    const y = 1.75 + i * 0.98;
    num(sl, 8.6, y, n, "note " + n);
    T(sl, h, { x: 9.15, y: y - 0.02, w: 3.45, h: 0.35, fontSize: 14, bold: true }, "note title " + n);
    T(sl, d, { x: 9.15, y: y + 0.34, w: 3.45, h: 0.6, fontSize: 12, color: "4A4A4A" }, "note text " + n);
  });
  note(sl, foot, 6.68);
  sl.addNotes(notes);
}
const L = (t, c) => ({ t, c });
codeSlide("03 — Code: pass-through", "Local projections: one regression per horizon",
  [L("def lp(df, y, oil, p=6, H=24):"),
   L("    d = pd.DataFrame({'Y': df[y]}); d['dY'] = d.Y.diff()"),
   L("    # (1) controls: lags of oil and of inflation", 1),
   L("    X = [oil.rename('oil')]"),
   L("    X += [oil.shift(j).rename(f'oil_l{j}') for j in range(1, p+1)]"),
   L("    X += [d.dY.shift(j).rename(f'dy_l{j}') for j in range(1, p+1)]"),
   L("    X = pd.concat(X, axis=1); out = []"),
   L("    for h in range(H + 1):              # (2) horizons 0..24", 1),
   L("        lhs = d.Y.shift(-h) - d.Y.shift(1)   # (3) pi(t+h)-pi(t-1)", 1),
   L("        z = pd.concat([lhs.rename('lhs'), X], axis=1).dropna()"),
   L("        f = sm.OLS(z.lhs, sm.add_constant(z.drop(columns='lhs')))"),
   L("              .fit(cov_type='HAC',             # (4) Newey-West", 1),
   L("                   cov_kwds={'maxlags': h + 1})"),
   L("        out.append((h, 10 * f.params['oil'],   # (5) per +10% Brent", 1),
   L("                       10 * f.bse['oil']))"),
   L("    return pd.DataFrame(out, columns=['h', 'b', 'se'])")],
  [[1, "Controls", "6 lags of oil and of inflation changes absorb past dynamics"],
   [2, "Horizon loop", "One OLS per horizon, no VAR structure imposed (Jordà, 2005)"],
   [3, "Cumulative change", "π(t+h) − π(t−1): β(h) traces the full response path"],
   [4, "Robust errors", "Overlapping horizons → Newey–West with h+1 lags"],
   [5, "Scaling", "β × 10 = pp of inflation per +10% Brent, as on every chart"]],
  "Simplified excerpt from code/oil_passthrough_analysis.py (function lp).",
  "(1.5 min) The engine of the pass-through results. For each horizon h from 0 to 24 months we run a separate regression of the change in inflation between t-1 and t+h on this month's oil shock, controlling for six lags of oil and of inflation. Overlapping horizons mean we need Newey-West standard errors. The coefficient times 10 is the effect of a 10% oil rise; plotting it against h gives the response curves.");
codeSlide("04 — Code: identification", "Separating demand from supply shocks",
  [L("U = W[['oil', 'mkt']].dropna(); U -= U.mean()  # Brent, S&P 500"),
   L("P = np.linalg.cholesky(np.cov(U.T))    # (1) one valid split", 1),
   L(""),
   L("def impact(th):                         # (2) rotate: A = P Q(theta)", 1),
   L("    Q = np.array([[np.cos(th), -np.sin(th)],"),
   L("                  [np.sin(th),  np.cos(th)]])"),
   L("    A = P @ Q; A = A * np.sign(A[0])    # both shocks raise oil"),
   L("    ok = A[1, 0] > 0 and A[1, 1] < 0    # (3) demand: stocks up", 1),
   L("    return A if ok else None            #     supply: stocks down", 1),
   L(""),
   L("NARR = {'2008-10-10': ('demand',   -1), # (4) known episodes", 1),
   L("        '2020-03-13': ('specific', -1),"),
   L("        '2022-03-04': ('specific', +1),"),
   L("        '2026-03-06': ('specific', +1)}"),
   L("ok = [th for th in grid if impact(th) is not None"),
   L("                         and narrative_ok(th)]"),
   L("A = impact(np.mean(ok)); e = np.linalg.solve(A, U.T.values)  # (5)", 1),
   L("oil_demand, oil_specific = A[0, 0] * e[0], A[0, 1] * e[1]")],
  [[1, "Starting point", "Cholesky factor: one decomposition of the covariance"],
   [2, "All candidates", "Every rotation θ is an equally valid statistical split"],
   [3, "Sign restrictions", "Demand moves stocks up, supply moves them down"],
   [4, "Narrative restrictions", "Lehman, 2020 price war, Ukraine, Iran: named shock dominates"],
   [5, "Two components", "Each weekly oil return = demand part + oil-specific part"]],
  "Simplified excerpt from code/oil_passthrough_analysis.py (section F). Uhlig (2005); Antolín-Díaz & Rubio-Ramírez (2018).",
  "(1.5 min) Raw oil moves mix two stories, and we use the stock market to tell them apart. Oil and stocks up together: global demand. Oil up, stocks down: an oil-specific cost shock such as a war. Infinitely many splits satisfy these signs, so we narrow them with four weeks whose cause is documented, take the middle of the admissible set, and check results at both ends.");

// ================================================================== 6 HEADLINE VS CORE
pres.addSection({ title: "Results" });
s = page("05 — Result: pass-through", "Oil hits headline inflation fast; core barely moves", "Results");
img(s, "fig02_headline_vs_core", 0.7, 1.65, 10.0, 2.75, "Chart: headline vs core inflation response to a 10% Brent rise");
verdict(s, 10.95, 2.0, "H1 SUPPORTED", C.accent6, "H1");
T(s, "Response of YoY inflation to +10% Brent, 1990–2026, 90% bands", { x: 10.95, y: 2.4, w: 1.7, h: 1.4, fontSize: 11, color: C.accent4 }, "chart caption");
hline(s, 0.7, 5.42, 11.9, "stats rule");
stats(s, [["+0.55 pp", "US headline, 6–12 months", C.accent3], ["+0.42 pp", "Euro area headline, 6 months", C.accent3],
          ["+0.12 pp", "US core, 6 months", C.accent2], ["+0.05 pp", "Euro area core, 6 months", C.accent2]], 5.55);
s.addNotes("(1 min) A 10% rise in Brent raises US headline inflation by about half a point within six months; it peaks around month 11 and fades as the shock leaves the year-on-year comparison. The euro area is somewhat smaller and slower. Core moves only about a tenth of a point. H1 supported.");

// ================================================================== 7 OVER TIME
s = page("06 — Result: over time", "Since 2020, oil reaches core inflation, and Europe more", "Results");
img(s, "fig01_passthrough_by_sample", 0.7, 1.6, 7.4, 1.571, "Chart: pass-through before and after 2020");
vline(s, 8.45, 1.75, 4.8, "column divider");
verdict(s, 8.85, 1.85, "H4 SUPPORTED", C.accent6, "H4");
T(s, "Core pass-through after 12 months", { x: 8.85, y: 2.2, w: 3.8, h: 0.35, fontSize: 14, bold: true }, "H4 header");
stats(s, [["0.07 → 0.68", "US, pp: pre-2020 → 2020–26", C.accent2]], 2.6, 8.85, 3.8);
stats(s, [["0.03 → 0.40", "Euro area, pp", C.accent2]], 3.75, 8.85, 3.8);
hline(s, 8.85, 4.95, 3.8, "verdict rule");
verdict(s, 8.85, 5.15, "H2 REJECTED", C.accent3, "H2");
T(s, "US > euro area before 2020: fixed per-litre fuel taxes damp the % move at the pump. Reversed since 2020 (gas, weak euro).", { x: 8.85, y: 5.45, w: 3.8, h: 1.2, fontSize: 13 }, "H2 text");
s.addNotes("(1 min) Splitting the sample. Before 2020 core pass-through was essentially zero; since 2020 it is large in both regions. Caveat: about 80 months only, and 2021-22 mixes oil with reopening and supply chains. H2 rejected over the full sample: European fuel taxes are fixed per litre, so a 10% crude move is a smaller percentage at the pump. Since 2020 the euro area is larger, probably gas and the weak euro.");

// ================================================================== 8 MARKETS
s = page("07 — Result: markets", "Markets price oil into 5-year, not 5y5y, expectations", "Results");
img(s, "fig04_market_betas_by_period", 0.7, 1.6, 10.0, 2.857, "Chart: sensitivity of inflation expectations to oil by period");
verdict(s, 10.95, 2.0, "H3 SUPPORTED", C.accent6, "H3");
T(s, "bp per +10% Brent, weekly changes, S&P 500 control, 90% CI", { x: 10.95, y: 2.4, w: 1.7, h: 1.4, fontSize: 11, color: C.accent4 }, "chart caption");
hline(s, 0.7, 5.42, 11.9, "stats rule");
stats(s, [["+6.8 bp", "US 5y breakeven, 2004–2026", C.accent3], ["+5.4 bp", "Euro area 5y swap", C.accent3],
          ["≈ 0 bp", "5y5y forwards (US −0.4, EA +1.2)", C.accent2], ["+4 bp", "5y5y in 2021–23: the exception", C.accent1]], 5.55);
s.addNotes("(1 min) A 10% oil rise moves 5-year inflation compensation by 5-7 bp. The 5y5y forward does not move on average. The exception is 2021-23, when inflation was high and central banks looked behind the curve: even the 5y5y moved about 4 bp. H3 holds on average, with an important exception.");

// ================================================================== 9 2022 vs 2026
s = page("08 — Case study", "2022 vs 2026: a bigger shock, but anchored expectations", "Results");
img(s, "fig05_2022_vs_2026", 0.7, 1.6, 11.9, 3.333, "Chart: Brent and inflation expectations after the 2022 and 2026 shocks");
hline(s, 0.7, 5.42, 11.9, "stats rule");
stats(s, [["+32% → +33/39 bp", "2022 Ukraine: Brent → 5y5y (US/EA). Markets priced de-anchoring risk", C.accent3],
          ["+63% → −5/+5 bp", "2026 Iran / Hormuz: record monthly oil rise, 5y5y unchanged", C.accent2]], 5.55);
s.addNotes("(1 min) The two biggest recent supply shocks. In 2022 a 32% oil rise came with a 33-39 bp jump in long-term expectations. In March 2026 Brent rose 63% at its peak, its largest monthly gain on record, yet the 5y5y barely moved. Caveat: 2022 also had a European gas shock and a central bank seen as too loose.");

// ================================================================== 10 DEMAND VS SUPPLY
s = page("09 — Result: shock type", "It is not oil itself: it is why oil moves", "Results");
img(s, "fig08_market_by_shock_type", 0.7, 1.7, 7.6, 2.5, "Chart: inflation expectations after demand-driven vs oil-specific oil rises");
note(s, "bp per +10% Brent · weekly 2004–2026 · 90% CI", 4.8);
vline(s, 8.65, 1.75, 3.35, "column divider");
stats(s, [["+13 bp", "US 5y5y after a demand-driven oil rise", C.accent2]], 1.8, 9.05, 3.6);
stats(s, [["−1 bp", "US 5y5y after an oil-specific (supply) rise, not significant", C.accent3]], 3.35, 9.05, 3.6);
hline(s, 0.7, 5.35, 11.9, "takeaway rule");
T(s, [
  { text: "Supply shock = temporary cost shock: ", options: { bold: true } },
  { text: "headline up for about a year, long-term expectations unchanged.  " },
  { text: "Demand-driven rise = stronger economy: ", options: { bold: true } },
  { text: "markets revise long-term inflation. Robust across all admissible identifications (US 5y5y: demand +9.6 to +20 bp, supply −2.4 to −0.4 bp)." },
], { x: 0.7, y: 5.55, w: 11.9, h: 1.1, fontSize: 15 }, "takeaway");
s.addNotes("(1.5 min) Our most original result. With the decomposition from code slide 2: when oil rises because of a supply shock, long-term expectations do not move. When it rises with global demand, the US 5y5y rises 13 bp. This is the cash-flow vs discount-rate distinction. Caveat: the demand effect includes demand itself, and breakevens carry risk premia.");

// ================================================================== 11 OUT OF SAMPLE
s = page("10 — Result: out of sample", "A pre-2020 model predicts inflation after the 2026 shock", "Results");
img(s, "fig06_out_of_sample_2026", 0.7, 1.6, 7.4, 1.571, "Chart: out-of-sample inflation forecasts after March 2026");
vline(s, 8.45, 1.75, 4.8, "column divider");
stats(s, [["3.76% vs 3.7%", "US headline, Sep 2026: forecast vs actual", C.accent2]], 1.8, 8.85, 3.8);
stats(s, [["+1.1 pp", "Oil's contribution (vs Brent frozen at Feb 2026)", C.accent3]], 3.35, 8.85, 3.8);
stats(s, [["0.21 vs 0.88", "Forecast error (RMSE, pp): model vs no-shock path", C.accent2]], 4.9, 8.85, 3.8);
note(s, "Model estimated on 1990–2019, fed the actual Brent path from March 2026; it never sees inflation after February 2026.", 6.42);
s.addNotes("(1.5 min) We estimate the model only on 1990-2019, give it the actual oil path after March 2026, and let it predict inflation month by month without seeing actual inflation. It predicts 3.76% US headline in September against 3.7% actual. Without the oil shock inflation would have been about 2.6%, so oil explains about 1.1 pp. 2026 behaved like the pre-2020 regime, which is what markets priced. The euro area is under-predicted by about 0.8 pp at the end, probably gas and the euro.");

// ================================================================== 12 CONCLUSION
pres.addSection({ title: "Conclusion" });
s = page("11 — Conclusion", "Markets read oil shocks correctly, when they are supply-driven", "Conclusion", "CLOSING");
[["H1", "SUPPORTED", "Headline +0.4–0.55 pp, core +0.1 pp", C.accent6],
 ["H2", "REJECTED", "US larger before 2020; euro area larger since", C.accent3],
 ["H3", "SUPPORTED*", "5y5y flat, except 2021–23 and demand-driven shocks", C.accent6],
 ["H4", "SUPPORTED", "Core pass-through rose after 2020", C.accent6]].forEach(([k, v, d, col], i) => {
  const x = 0.7 + i * 2.975;
  if (i > 0) vline(s, x - 0.2, 1.85, 1.9, "verdict divider " + i, "3A3A3A");
  T(s, k, { x, y: 1.8, w: 1.0, h: 0.6, fontSize: 30, bold: true, color: C.background1 }, k + " label");
  s.addShape(pres.shapes.RECTANGLE, { x, y: 2.55, w: 0.14, h: 0.14, fill: { color: col === C.accent6 ? "4CB879" : "E8705F" }, line: { color: col === C.accent6 ? "4CB879" : "E8705F" }, objectName: k + " marker" });
  T(s, v, { x: x + 0.24, y: 2.48, w: 2.4, h: 0.3, fontSize: 12, bold: true, charSpacing: 1, color: col === C.accent6 ? "4CB879" : "E8705F" }, k + " verdict");
  T(s, d, { x, y: 2.9, w: 2.6, h: 0.9, fontSize: 14, color: C.accent5 }, k + " text");
});
hline(s, 0.7, 4.1, 11.9, "conclusion rule", "3A3A3A");
T(s, [
  { text: "KEY TAKEAWAY", options: { bold: true, color: C.accent1, charSpacing: 2, fontSize: 12, breakLine: true } },
  { text: "Oil-specific shocks raise headline inflation for about a year and leave long-term expectations anchored; markets price this correctly, as the 2026 out-of-sample test shows. Long-term expectations move only when oil rises with global demand, or when inflation is already high.", options: { color: C.background1 } },
], { x: 0.7, y: 4.35, w: 7.2, h: 2.3, fontSize: 16, paraSpaceAfter: 6 }, "takeaway");
T(s, [
  { text: "LIMITATIONS", options: { bold: true, color: C.accent1, charSpacing: 2, fontSize: 12, breakLine: true } },
  { text: "Oil not fully exogenous", options: { bullet: true, breakLine: true } },
  { text: "Breakevens include risk and liquidity premia", options: { bullet: true, breakLine: true } },
  { text: "Short post-2020 sample", options: { bullet: true, breakLine: true } },
  { text: "No gas or EUR/USD for the euro area", options: { bullet: true } },
], { x: 8.4, y: 4.35, w: 4.2, h: 2.3, fontSize: 14, color: C.accent5, paraSpaceAfter: 4 }, "limitations");
s.addNotes("(2 min) Summarize the four verdicts. Markets do not react to oil itself but to its cause: supply shocks are treated as temporary, and the 2026 out-of-sample test suggests markets were right. Limitations: endogeneity of oil, risk premia in breakevens, short post-2020 sample, no gas or exchange-rate data for Europe. Then questions.");

(async () => {
  const out = path.join(__dirname, "Oil_shocks_inflation_presentation.pptx");
  await pres.writeFile({ fileName: out });
  if (applyTheme) await applyTheme(out, THEME);
  console.log("written", out);
})();
