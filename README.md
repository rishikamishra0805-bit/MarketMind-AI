# MarketMind AI
**Intelligent Customer Segmentation & Recommendation System**

A beginner-friendly machine-learning project that analyses retail transactions, builds RFM (Recency, Frequency, Monetary) customer features, groups customers with K-Means, and displays product suggestions based on purchase history and group popularity.

## Features
- Retail CSV upload and included synthetic demo data
- Data cleaning and RFM feature engineering
- K-Means customer segmentation with StandardScaler
- Silhouette score for cluster evaluation
- Interactive Plotly charts and segment profiles
- Simple item suggestions using customer history and segment popularity
- CSV downloads for customer-level and segment-level results

## Run locally (Windows)
1. Install Python 3.11 or 3.12 from https://www.python.org/downloads/
2. Open this folder in VS Code.
3. Open Terminal > New Terminal.
4. Create a virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   ```
5. Activate it:
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.
6. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
7. Start the app:
   ```powershell
   streamlit run app.py
   ```

## Dataset format
The app expects transaction-level CSV columns:
`InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID`
`Country` is optional.

The included `data/sample_transactions.csv` is **synthetic demo data** created to test the app. It is not real customer information and should not be used to claim real-world model performance.

For the final academic project, download the UCI Online Retail dataset from:
https://archive.ics.uci.edu/dataset/352/online+retail

Upload the downloaded CSV through the app. The UCI file may be named `Online Retail.xlsx`; save/export it as CSV if needed.

## Methodology
1. Clean invalid dates, missing customer IDs, cancelled invoices, returns and non-positive prices/quantities.
2. Calculate:
   - Recency: days since the customer's latest purchase
   - Frequency: number of distinct invoices
   - Monetary: total spend
3. Apply `log1p` and StandardScaler to reduce skew and standardize the features.
4. Train K-Means and compare cluster profiles. Use the silhouette score as one diagnostic, not as proof that the clusters are commercially correct.
5. Generate simple product suggestions from the customer's previous purchases and popular items in the same cluster.

## Limitations / responsible use
- This is a decision-support demo, not a system that knows a customer's intentions.
- Cluster numbers are arbitrary; cluster 0 is not inherently better than cluster 1.
- The recommendation component is a transparent popularity/history baseline, not a trained deep-learning recommender.
- Use anonymized/public data and do not upload private customer data to a public deployment.

## Project structure
```text
MarketMind_AI/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── sample_transactions.csv
├── models/
├── reports/
└── src/
    ├── data_utils.py
    ├── segmentation.py
    └── recommendations.py
```

## Future enhancements
- RFM-based human-readable segment naming
- Elbow and silhouette comparison for multiple K values
- Association-rule recommendations (Apriori)
- Model persistence and automated report generation
- Unit tests and project documentation
