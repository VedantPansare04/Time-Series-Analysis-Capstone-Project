# TSA Capstone Project — Time Series Analysis & Portfolio Construction

## Objective
End-to-end time series analysis of 5 NSE-listed stocks from diversified sectors, covering data collection, preprocessing, stationarity testing, ARIMA & Holt-Winters forecasting, volatility analysis, STL decomposition, portfolio construction, and model comparison.

## Stocks Analyzed
| Stock | Sector |
|-------|--------|
| TCS.NS | IT |
| HDFCBANK.NS | Banking |
| RELIANCE.NS | Energy |
| SUNPHARMA.NS | Pharma |
| TATAMOTORS.NS | Auto |

## Setup
```bash
pip install -r requirements.txt
python tsa_capstone_main.py
```

## Directory Structure
```
tsa_capstone/
├── data/raw/ processed/ returns/ forecasts/
├── outputs/plots/ metrics/ decomposition/ forecasts/ portfolio/
├── report/figures/ final_report.md
├── tsa_capstone_main.py
├── requirements.txt
└── README.md
```

## Date Range
- **Training**: 2021-01-01 to 2025-06-30
- **Testing**: 2025-07-01 to 2025-12-31
- **Capital**: ₹10,00,000
