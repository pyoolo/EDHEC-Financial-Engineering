# Data

The raw data are **not included** in this repository: they come from Bloomberg, whose terms do not allow redistribution.

To reproduce the results, put the Bloomberg download file here as:

```
data/Bloomberg_download_oil_inflation.xlsx
```

The workbook needs two sheets in Bloomberg `BDH` layout (blocks of three columns: date, value, blank; row 1 = series label, row 2 = ticker):

| Sheet | Frequency | Series (label used by the code → ticker) |
|---|---|---|
| `MENSILE` | Monthly, 1990– | Brent (front month) → `CO1 Comdty`; WTI (front month) → `CL1 Comdty`; Bloomberg Commodity Index → `BCOM Index`; US CPI headline YoY → `CPI YOY Index`; US CPI core YoY → `CPI XYOY Index`; US CPI livello indice (NSA) → `CPURNSA Index`; Euro HICP headline YoY → `ECCPEMUY Index`; Euro HICP core YoY → `CPEXEMUY Index`; Euro HICP livello indice → `CPTFEMU Index` |
| `GIORNALIERO` | Daily, 2004– | Brent (front month) → `CO1 Comdty`; WTI (front month) → `CL1 Comdty`; US 5y breakeven → `USGGBE05 Index`; US 5y5y inflation swap fwd → `FWISUS55 Index`; Euro 5y inflation swap → `EUSWI5 Index`; Euro 5y5y inflation swap fwd → `FWISEU55 Index`; S&P 500 Energy → `S5ENRS Index`; STOXX 600 Oil & Gas → `SXEP Index`; S&P 500 → `SPX Index` |

The code finds series by the **label in row 1**, so keep those labels (including the trailing space in `"S&P 500 "`) if you rebuild the file.

Free backups for the US series are available on FRED (CPIAUCSL, CPILFESL, T5YIE, T5YIFR).
