#!/usr/bin/env bash
# Rebuilds every result, table, figure, the paper and the presentation images from the Bloomberg file.
# Usage: ./run_all.sh [path/to/Bloomberg_download_oil_inflation.xlsx]
set -euo pipefail
cd "$(dirname "$0")"
DATA="${1:-data/Bloomberg_download_oil_inflation.xlsx}"

echo "1/4  Analysis (about 30 seconds)"
python3 code/oil_passthrough_analysis.py "$DATA" output

echo "2/4  LaTeX tables and figures"
python3 code/make_tables.py output/results_tables.xlsx latex/tables
cp output/fig*.pdf latex/figures/

echo "3/4  Paper"
if command -v latexmk >/dev/null; then
  (cd latex && latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex >/dev/null && cp paper.pdf ../paper/Oil_shocks_inflation_passthrough_paper.pdf && latexmk -c >/dev/null)
else
  echo "     latexmk not found: compile latex/paper.tex yourself (or upload the latex/ folder to Overleaf)"
fi

echo "4/4  Chart images for the presentation"
if command -v pdftoppm >/dev/null; then
  for f in fig01_passthrough_by_sample fig02_headline_vs_core fig04_market_betas_by_period fig05_2022_vs_2026 fig06_out_of_sample_2026 fig08_market_by_shock_type; do
    pdftoppm -png -r 220 -singlefile "output/$f.pdf" "presentation/img/$f"
  done
  echo "     rebuild the deck with: cd presentation && npm install && npm run build"
else
  echo "     pdftoppm not found: presentation images not refreshed"
fi
echo "Done."
