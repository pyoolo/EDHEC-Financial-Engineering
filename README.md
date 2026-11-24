# EDHEC Financial Engineering — 2026-2027

Course projects, assignments, notebooks and lecture notes from my year at **EDHEC Business School** (Financial Engineering).

**Author:** Paolo Mastrogiacomo ([@pyoolo](https://github.com/pyoolo))

---

## Repository structure

| Folder | Content |
|---|---|
| [`Advanced Derivatives`](Advanced%20Derivatives) | Notes and projects on derivatives pricing |
| [`Advanced Fixed Income & Credit`](Advanced%20Fixed%20Income%20%26%20Credit) | Notes and projects on fixed income and credit |
| [`Introduction To Machine Learning`](Introduction%20To%20Machine%20Learning) | Machine learning projects and notebooks |
| [`Macro & Capital Markets`](Macro%20%26%20Capital%20Markets) | Notes and projects on macroeconomics and capital markets |
| [`Quantitative Portfolio Management`](Quantitative%20Portfolio%20Management) | Assignments, teaching notebooks and LaTeX lecture notes (Prof. Raman Uppal) |
| [`Research-Thesis`](Research-Thesis) | Research thesis |

### Quantitative Portfolio Management

```
Quantitative Portfolio Management/
├── assignments/
│   ├── Assignment-1/   # Financial data and performance measurement
│   ├── Assignment-2/   # Mean-variance portfolios and estimation error
│   └── Assignment-3/   # Black-Litterman model
├── notebooks/          # Didactic Python notebooks, one per lecture topic
├── notes/              # LaTeX lecture notes (Part 1: Ch. 1-4, Part 2: Ch. 5-8)
├── README.md
└── requirements.txt
```

Topics: estimation of means and covariances, out-of-sample performance evaluation, Markowitz portfolios, shrinkage and constrained portfolios, Black–Litterman, factor models and parametric portfolio policies, volatility-managed portfolios, systematic and unsystematic risk.

---

## Getting started

```bash
git clone https://github.com/pyoolo/EDHEC-Financial-Engineering.git
cd EDHEC-Financial-Engineering
pip install -r "Quantitative Portfolio Management/requirements.txt"
```

- **Notebooks** (`.ipynb`): open in Jupyter or Google Colab and run *Restart & Run All*. Market data are downloaded with `yfinance`, so an internet connection is needed.
- **LaTeX notes** (`.tex`): compile with `pdflatex` (twice, for the table of contents) or upload to Overleaf.

## Tools

Python (NumPy, pandas, SciPy, Matplotlib, yfinance) · Jupyter / Colab · LaTeX

---

## Disclaimer

This repository is for personal study. Course slides and assignment texts belong to their authors and EDHEC Business School and are not redistributed here. Assignments were completed in groups; group members are credited inside each notebook.
