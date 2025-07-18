import streamlit as st
import polars as pl
import os
from dotenv import load_dotenv

load_dotenv()

MCP_FILESYSTEM_DIR = os.environ.get("MCP_FILESYSTEM_DIR", "./data")

st.title("Current Financial Data")

def display_dataframe(title: str, parquet_path: str, is_series: bool = False):
    if os.path.exists(parquet_path):
        try:
            df = pl.read_parquet(parquet_path)
            if is_series:
                df = df.to_series(0).to_frame("target")

            st.subheader(title)
            show_full = st.checkbox(f"Show full {title.lower()}", key=f"full_{title.replace(' ', '_')}")

            if not show_full:
                n_rows = st.number_input(f"Number of rows to show for {title.lower()}", min_value=1, value=10, key=f"rows_{title.replace(' ', '_')}")
                display_df = df.head(n_rows)
            else:
                display_df = df
            st.dataframe(display_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading data: {e}")
    else:
        st.info("No data loaded yet.")

data_parquet = os.path.join(MCP_FILESYSTEM_DIR, "default_finance_session_data.parquet")  # Assumes default user_id; update if dynamic
display_dataframe("General Data", data_parquet)

transactions_parquet = os.path.join(MCP_FILESYSTEM_DIR, "default_finance_session_transactions.parquet")
display_dataframe("Transactions", transactions_parquet)

budgets_parquet = os.path.join(MCP_FILESYSTEM_DIR, "default_finance_session_budgets.parquet")
display_dataframe("Budgets", budgets_parquet)

savings_goals_parquet = os.path.join(MCP_FILESYSTEM_DIR, "default_finance_session_savings_goals.parquet")
display_dataframe("Savings Goals", savings_goals_parquet)

investments_parquet = os.path.join(MCP_FILESYSTEM_DIR, "default_finance_session_investments.parquet")
display_dataframe("Investments", investments_parquet)
