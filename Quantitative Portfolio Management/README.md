# Quantitative Portfolio Management (EDHEC, 2026–2027)

Course by Prof. Raman Uppal. Lecture notes, teaching notebooks and assignment solutions
for **Part 1** of the course (Lectures 1–4).

| Folder | Content |
|---|---|
| `notes/` | `QPM_Notes_Part1.tex` / `.pdf` – extended lecture notes, Chapters 0–4: syllabus, theory with derivations, worked examples, Python exercises, self-tests, assignment guides, formula sheet, English–Italian glossary. Single LaTeX file (standard packages only; compiles on Overleaf). |
| `notebooks/` | `QPM_00_Python_Toolkit` (NumPy/pandas for portfolio management, exam-oriented) and `QPM_01_Managing_Financial_Data` (Chapter 1 applied, with simulations and real data). |
| `assignments/Assignment-1/` | Lectures 1–2: FAANG daily returns (moments, skewness, kurtosis, Normality tests); S&P 500 monthly returns, annualized mean/volatility/Sharpe ratio. Saves `sp500_monthly_returns.csv`. |
| `assignments/Assignment-2/` | Lectures 3–4: rolling-window (60 months) out-of-sample backtest of MVP, GMV, MVP-C and GMV-C on 10 S&P 500 stocks; Sharpe ratios, turnover, Jobson–Korkie–Memmel tests. |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook
```

or open the notebooks in Google Colab (uncomment the `!pip install` line in the first code cell).

* Run **Assignment 1 first**: it downloads the data (Wikipedia, Yahoo! Finance, Ken French data library)
  and saves `sp500_monthly_returns.csv`. Assignment 2 reads that file from its own folder or from
  `../Assignment-1/`, and rebuilds it if it is not found.
* Downloading ~500 tickers takes a few minutes; tickers that fail are ignored, as the assignment allows.
* Results depend on the download date (Yahoo rescales adjusted prices after splits and dividends).

## Before submitting an assignment

1. Fill in Section 0 (date, group members in alphabetical order with LAST NAMES in capitals, comments).
2. Rename the file `QPM-2026-Assignment-n-A-B-C-D.ipynb` with the group's last names.
3. *Kernel → Restart Kernel and Run All Cells* (Colab: *Runtime → Restart session and run all*): no errors.
4. Email to the instructor with subject `QPM 2026-2027: Assignment n` before 22:00 CET on the due date
   (Assignment 1: 11 Nov; Assignment 2: 16 Nov).

Course slides and assignment PDFs are not included (course material).
