import pandas as pd
from pathlib import Path


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA = BASE_DIR / "data" / "processed" / "telco_churn_clean.csv"


def load_data():
    """Load the raw Telco Customer Churn dataset."""
    df = pd.read_csv(RAW_DATA)
    return df


def clean_data(df):
    """Clean and preprocess the dataset."""

    # Remove leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    # Remove rows with missing values
    df = df.dropna()

    # Remove duplicate customer records
    df = df.drop_duplicates()

    return df


def main():
    df = load_data()

    print("Original dataset:")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nMissing values:")
    print(df.isnull().sum())

    df_clean = clean_data(df)

    print("\nClean dataset:")
    print(f"Rows: {df_clean.shape[0]}")
    print(f"Columns: {df_clean.shape[1]}")

    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROCESSED_DATA, index=False)

    print(f"\nClean dataset saved to: {PROCESSED_DATA}")


if __name__ == "__main__":
    main()