
import re
import pandas as pd

REQUIRED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID"
]


def load_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean retail transaction data before customer analysis."""

    df = df.copy()

    # Standardize column names.
    rename = {}

    column_mapping = {
        "invoiceno": "InvoiceNo",
        "stockcode": "StockCode",
        "description": "Description",
        "quantity": "Quantity",
        "invoicedate": "InvoiceDate",
        "unitprice": "UnitPrice",
        "customerid": "CustomerID",
        "country": "Country"
    }

    for col in df.columns:
        key = re.sub(r"[\s_-]+", "", str(col).strip().lower())

        if key in column_mapping:
            rename[col] = column_mapping[key]

    df = df.rename(columns=rename)

    # Check required columns.
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    # Convert columns into correct data types.
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce",
        dayfirst=True
    )

    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    )

    df["UnitPrice"] = pd.to_numeric(
        df["UnitPrice"],
        errors="coerce"
    )

    # Clean customer IDs such as 12347.0 into 12347.
    df["CustomerID"] = (
        df["CustomerID"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
    )

    # Clean invoice numbers.
    df["InvoiceNo"] = (
        df["InvoiceNo"]
        .astype("string")
        .str.strip()
    )

    # Remove records with missing important values.
    df = df.dropna(
        subset=[
            "InvoiceDate",
            "Quantity",
            "UnitPrice",
            "CustomerID",
            "InvoiceNo"
        ]
    )

    # Remove empty customer IDs.
    df = df[df["CustomerID"] != ""]

    # Remove cancelled invoices.
    df = df[
        ~df["InvoiceNo"].str.upper().str.startswith("C", na=False)
    ]

    # Remove returns and invalid prices or quantities.
    df = df[
        (df["Quantity"] > 0) &
        (df["UnitPrice"] > 0)
    ]

    # Calculate transaction revenue.
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # Clean product descriptions.
    df["Description"] = (
        df["Description"]
        .fillna("Unknown item")
        .astype(str)
        .str.strip()
    )

    df.loc[
        df["Description"] == "",
        "Description"
    ] = "Unknown item"

    # Reset row numbering.
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No valid transactions found after data cleaning."
        )

    return df


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Recency, Frequency and Monetary for each customer."""

    if df.empty:
        raise ValueError(
            "Cannot calculate RFM because the dataset is empty."
        )

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=(
            "InvoiceDate",
            lambda dates: (snapshot_date - dates.max()).days
        ),
        Frequency=(
            "InvoiceNo",
            "nunique"
        ),
        Monetary=(
            "Revenue",
            "sum"
        )
    ).reset_index()

    return rfm
