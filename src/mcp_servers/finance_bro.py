import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from mcp.server.fastmcp import FastMCP
from typing import Optional, Dict, Union, List, Any
import os

import joblib
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("finance_wizard")


class FinanceBro:

    def __init__(self, user_id: str = "default"):  # Added user_id for multi-user persistence
        self.data: Optional[pl.DataFrame] = None


        self.transactions: Optional[pl.DataFrame] = None
        self.budgets: Optional[pl.DataFrame] = None
        self.savings_goals: Optional[pl.DataFrame] = None
        self.investments: Optional[pl.DataFrame] = None

        self.working_dir = os.environ.get("MCP_FILESYSTEM_DIR")
        if self.working_dir:
            os.makedirs(self.working_dir, exist_ok=True)
        self.session_base = os.path.join(self.working_dir, f"{user_id}_finance_session") if self.working_dir else None

    def _save_data(self) -> None:
        if self.session_base and self.data is not None:
            try:
                data_parquet = self.session_base + "_data.parquet"
                self.data.write_parquet(data_parquet)


                if self.transactions is not None:
                    transactions_parquet = self.session_base + "_transactions.parquet"
                    self.transactions.write_parquet(transactions_parquet)
                if self.budgets is not None:
                    budgets_parquet = self.session_base + "_budgets.parquet"
                    self.budgets.write_parquet(budgets_parquet)
                if self.savings_goals is not None:
                    savings_goals_parquet = self.session_base + "_savings_goals.parquet"
                    self.savings_goals.write_parquet(savings_goals_parquet)
                if self.investments is not None:
                    investments_parquet = self.session_base + "_investments.parquet"
                    self.investments.write_parquet(investments_parquet)

                session_joblib = self.session_base + ".joblib"
                state = {}
                joblib.dump(state, session_joblib)

            except Exception as e:
                print(f"Warning: Failed to save: {e}")

    def _load_data_from_session(self) -> bool:
        if self.session_base:
            session_joblib = self.session_base + ".joblib"
            if os.path.exists(session_joblib):
                try:
                    data_parquet = self.session_base + "_data.parquet"
                    self.data = pl.read_parquet(data_parquet) if os.path.exists(data_parquet) else None

                    transactions_parquet = self.session_base + "_transactions.parquet"
                    self.transactions = pl.read_parquet(transactions_parquet) if os.path.exists(
                        transactions_parquet) else None
                    budgets_parquet = self.session_base + "_budgets.parquet"
                    self.budgets = pl.read_parquet(budgets_parquet) if os.path.exists(budgets_parquet) else None
                    savings_goals_parquet = self.session_base + "_savings_goals.parquet"
                    self.savings_goals = pl.read_parquet(savings_goals_parquet) if os.path.exists(
                        savings_goals_parquet) else None
                    investments_parquet = self.session_base + "_investments.parquet"
                    self.investments = pl.read_parquet(investments_parquet) if os.path.exists(
                        investments_parquet) else None

                    state = joblib.load(session_joblib)
                    return True
                except Exception as e:
                    print(f"Warning: Failed to load session data: {e}")
                    return False
        return False

    async def load_data(self, file_path: str) -> str:
        if not self.working_dir:
            return "ERROR: MCP_FILESYSTEM_DIR environment variable not set."

        full_path = os.path.join(self.working_dir, file_path)
        try:
            if file_path.lower().endswith('.csv'):
                self.data = pl.read_csv(full_path, infer_schema_length=10000)
            elif file_path.lower().endswith('.xlsx'):

                self.data = pl.read_excel(full_path, sheet_id=None)
            else:
                return f"Error: Unsupported file format for {file_path}. Only .csv and .xlsx are supported."

            self.transactions = self.data.clone()

            self._save_data()
            return f"Data loaded from {full_path}"
        except Exception as e:
            return f"Error loading data from {full_path}: {e}"

    async def reset_session(self) -> str:
        """
        Reset the current session by deleting the persisted data file and clearing in-memory data
        """
        if self.session_base:
            session_joblib = self.session_base + ".joblib"
            if os.path.exists(session_joblib):
                try:
                    os.remove(session_joblib)
                    # Remove Parquet files
                    data_parquet = self.session_base + "_data.parquet"
                    if os.path.exists(data_parquet):
                        os.remove(data_parquet)
                    transactions_parquet = self.session_base + "_transactions.parquet"
                    if os.path.exists(transactions_parquet):
                        os.remove(transactions_parquet)
                    budgets_parquet = self.session_base + "_budgets.parquet"
                    if os.path.exists(budgets_parquet):
                        os.remove(budgets_parquet)
                    savings_goals_parquet = self.session_base + "_savings_goals.parquet"
                    if os.path.exists(savings_goals_parquet):
                        os.remove(savings_goals_parquet)
                    investments_parquet = self.session_base + "_investments.parquet"
                    if os.path.exists(investments_parquet):
                        os.remove(investments_parquet)
                except Exception as e:
                    return f"Error resetting session: {e}"

        if self.working_dir:
            try:
                for file in os.listdir(self.working_dir):
                    if file.endswith((".csv", ".xlsx")):
                        os.remove(os.path.join(self.working_dir, file))
            except Exception as e:
                return f"Error deleting .csv/.xlsx files during reset: {e}"

        self.data = None
        self.transactions = None
        self.budgets = None
        self.savings_goals = None
        self.investments = None
        return "Session reset successfully. You can now load new data."

    async def save_to_csv(self, file_path: str) -> str:
        """
        Saves the current state of the DataFrame to a .csv file in the working directory.

        Args:
            file_path: The name for the output .csv file.
        """
        if self.data is None:
            # Try to load from session as a fallback
            if not self._load_data_from_session():
                return "Error: No data in memory to save. Please load data first."

        if not self.working_dir:
            return "ERROR: MCP_FILESYSTEM_DIR environment variable not set."

        if not file_path.lower().endswith('.csv'):
            file_path += '.csv'

        full_path = os.path.join(self.working_dir, file_path)
        if os.path.exists(full_path):
            return f"Error: File already exists at {full_path}. Choose a different name to avoid overwriting."

        try:
            self.data.write_csv(full_path)
            return f"Dataframe successfully saved to {full_path}. The task is complete. Please ask the user for the next action."
        except Exception as e:
            return f"Error: Failed to save data to {full_path}: {e}"

    async def inspect_data(self, n_rows: int = 5) -> str:
        """
        Initial data inspection. Includes first n rows of data, DataFrame info,
        data types of each column, missing values per column, duplicate rows,
        number of unique values per column.
        """
        if self.data is None:
            if not self._load_data_from_session():
                return "Error: No data loaded. Please load data first using 'load_data'."

        df_head = self.data.head(n_rows).to_dicts()

        describe_str = str(self.data.describe())

        schema_str = str(self.data.schema)

        null_values = self.data.null_count().to_dicts()[0]

        duplicate_rows = self.data.is_duplicated().sum()

        unique_columns = {col: self.data.select(pl.col(col).n_unique()).item() for col in self.data.columns}

        analysis_report = f"""
        DataFrame Analysis Report
        --- FIRST N ROWS ---
        {df_head}

        --- DESCRIBE ---
        {describe_str}

        --- SCHEMA (DATA TYPES) ---
        {schema_str}

        --- MISSING VALUES ---
        {null_values}

        --- DUPLICATE ROWS ---
        Found {duplicate_rows} duplicate rows.

        --- UNIQUE VALUES OF EVERY COLUMN ---
        {unique_columns}

        """
        return analysis_report.strip()

    async def plot_distribution(self, column_name: str, bins: Optional[int] = None) -> str:
        """
        Generates a histogram to visualize the distribution of a selected numerical column

        Args:
            column_name: The name of the column to plot (must be numerical)
            bins (Optional): Number of bins for the histogram (defaults to auto if None)

        """
        if self.data is None:
            if not self._load_data_from_session():
                return "Error: No data loaded. Please load data first using 'load_data'."

        if column_name not in self.data.columns:
            return f"Error: Column '{column_name}' not found in the dataset."

        col_dtype = self.data.schema[column_name]
        if col_dtype not in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32,
                             pl.UInt16, pl.UInt8):
            return f"Error: Column '{column_name}' is not numerical (type: {col_dtype}). Distributions only allow numerical columns."
        try:
            col_series = self.data.select(pl.col(column_name)).to_series().to_numpy()

            plt.figure()
            plt.hist(col_series, bins='auto' if bins is None else bins)
            plt.title(f"Distribution of {column_name}")
            plt.xlabel(column_name)
            plt.ylabel("Frequency")

            plot_path = os.path.join(self.working_dir, f"distribution_{column_name}.png")
            plt.savefig(plot_path)
            plt.close()

            return f"Distribution for '{column_name}' generated and saved at {plot_path}."

        except Exception as e:
            return f"Error: Failed to generate distribution for {column_name}: {e}"

    async def plot_scatter(self, x_column: str, y_column: str, hue_column: Optional[str] = None) -> str:
        """
        Generates a scatter plot to visualize the relationship between two columns.
        Optionally colors points by a third categorical column (hue).

        Args:
            x_column: The name of the column for the x-axis (numerical or categorical).
            y_column: The name of the column for the y-axis (numerical or categorical).
            hue_column (Optional): The name of a categorical column to color points by.
        """
        if self.data is None:
            if not self._load_data_from_session():
                return "Error: No data loaded. Please load data first using 'load_data'."

        missing_cols = [col for col in [x_column, y_column, hue_column] if col and col not in self.data.columns]
        if missing_cols:
            return f"Error: Column(s) {missing_cols} not found in the dataset."

        try:
            x_data = self.data.select(pl.col(x_column)).to_series().to_numpy()
            y_data = self.data.select(pl.col(y_column)).to_series().to_numpy()

            plt.figure()
            if hue_column:
                hue_data = self.data.select(pl.col(hue_column)).to_series().to_numpy()
                unique_hues = np.unique(hue_data)
                colors = plt.cm.get_cmap('viridis', len(unique_hues))  # Use a colormap for categories
                color_map = {hue: colors(i) for i, hue in enumerate(unique_hues)}
                for hue in unique_hues:
                    mask = (hue_data == hue)
                    plt.scatter(x_data[mask], y_data[mask], label=str(hue), color=color_map[hue])
                plt.legend(title=hue_column)
                file_name = f"scatter_{x_column}_{y_column}_hue_{hue_column}.png"
            else:
                plt.scatter(x_data, y_data)
                file_name = f"scatter_{x_column}_{y_column}.png"

            plt.title(f"Scatter Plot: {x_column} vs {y_column}" + (f" (Hue: {hue_column})" if hue_column else ""))
            plt.xlabel(x_column)
            plt.ylabel(y_column)

            plot_path = os.path.join(self.working_dir, file_name)
            plt.savefig(plot_path)
            plt.close()

            return f"Scatter plot for '{x_column}' vs '{y_column}'{f' (hue: {hue_column})' if hue_column else ''} generated and saved at {plot_path}."

        except Exception as e:
            return f"Error: Failed to generate scatter plot: {e}"

    async def impute_missing_values(self, imputation_map: Dict[str, str]) -> str:
        """
        Replaces missing values in a column by using a descriptive statistic (mean, median, or mode).

        Args:
            imputation_map: A dictionary that maps column names with imputation strategy.
        """
        if self.data is None:
            if not self._load_data_from_session():
                return "Error: No data loaded. Please load data first using 'load_data'."

        if not imputation_map:
            return "No imputation map provided; no changes made."

        allowed_strategies = ["mean", "median", "mode"]
        successful_imputations = []
        errors = []

        for column_name, imputation_strat in imputation_map.items():

            if column_name not in self.data.columns:
                errors.append(f"Error: Column '{column_name}' not found in the dataset.")
                continue

            if imputation_strat not in allowed_strategies:
                errors.append(
                    f"Invalid strategy '{imputation_strat}' for column '{column_name}'. Valid strategies are {allowed_strategies}")
                continue

            try:
                if imputation_strat in ["mean", "median", "mode"]:

                    if imputation_strat == "mean":
                        fill_value = self.data.select(pl.col(column_name).mean()).item()
                    elif imputation_strat == "median":
                        fill_value = self.data.select(pl.col(column_name).median()).item()
                    else:  # mode
                        fill_value = self.data.select(pl.col(column_name).mode().first()).item()
                    self.data = self.data.with_columns(pl.col(column_name).fill_null(fill_value))

                    successful_imputations.append(
                        f"Successfully imputed column '{column_name}' with strategy '{imputation_strat}'.")

            except Exception as e:
                errors.append(f"Failed to impute column '{column_name}' with strategy '{imputation_strat}': {e}.")

        self._save_data()

        if len(successful_imputations) == len(imputation_map) and len(errors) == 0:
            imputation_report = f"""
            Successfully performed imputation for missing values in columns with no errors: {list(imputation_map.keys())}.

            --- SUCCESSFUL IMPUTATIONS ---
            {'\n'.join(successful_imputations)}

            """
        else:
            imputation_report = f"""
            Unsuccessfully performed imputation for missing values in columns: {list(imputation_map.keys())}.

            --- UNSUCCESSFUL IMPUTATIONS ---
            {'\n'.join(errors)}

            --- SUCCESSFUL IMPUTATIONS ---
            {'\n'.join(successful_imputations)}

            """

        return imputation_report.strip()

    async def convert_to_numeric(self, column_name: str) -> str:
        """
        Converts values in a column to a numeric type (Float64). If values cannot be converted, set to null.

        Args:
            column_name: The name of the column to convert.
        """

        if self.data is None:
            if not self._load_data_from_session():
                return "Error: No data loaded. Please load data first using 'load_data'."

        column_name = column_name.strip()
        if not column_name:
            return "Error: Column name cannot be empty."

        if column_name not in self.data.columns:
            return f"Error: Column '{column_name}' not found in the dataset."

        try:

            self.data = self.data.with_columns(
                pl.col(column_name).cast(pl.Float64, strict=False).alias(column_name)
            )

            new_type = self.data.schema[column_name]


        except pl.exceptions.ComputeError as e:
            return f"Error: Column '{column_name}' could not be converted to numeric type: {e}"
        except Exception as e:
            return f"Unexpected error during conversion: {e}"

        self._save_data()

        return f"Column '{column_name}' successfully converted to numeric type ({new_type})."

    async def detect_outliers(self, outlier_map: Dict[str, str], z_score_threshold: float = 3.0) -> str:
        """
        Detects outliers in numerical columns using IQR or Z-score and returns a report.

        Args:
            outlier_map: A dictionary mapping column names to the detection strategy ('iqr' or 'z_score').
            z_score_threshold: The threshold for the Z-score method (defaults to 3.0).
        """
        if self.data is None:
            if not self._load_data_from_session():
                return "Error: No data loaded. Please load data first using 'load_data'."

        if not outlier_map:
            return "No outlier map provided; no outlier detection performed."

        report_lines = []
        errors = []

        for column_name, strategy in outlier_map.items():
            if column_name not in self.data.columns:
                errors.append(f"Error: Column '{column_name}' not found.")
                continue

            try:
                total_rows = self.data.shape[0]
                outliers = None

                if strategy == "iqr":
                    q1 = self.data[column_name].quantile(0.25)
                    q3 = self.data[column_name].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = self.data.filter(
                        (pl.col(column_name) < lower_bound) | (pl.col(column_name) > upper_bound))

                elif strategy == "z_score":
                    mean = self.data[column_name].mean()
                    std = self.data[column_name].std()
                    z_scores = (self.data[column_name] - mean) / std
                    outliers = self.data.filter(z_scores.abs() > z_score_threshold)

                else:
                    errors.append(f"Invalid strategy '{strategy}' for column '{column_name}'.")
                    continue

                num_outliers = outliers.shape[0]
                percentage = (num_outliers / total_rows) * 100 if total_rows > 0 else 0
                report_lines.append(
                    f"Column '{column_name}' ({strategy}): Found {num_outliers} outliers ({percentage:.2f}%)."
                )

            except Exception as e:
                errors.append(f"Failed to detect outliers in column '{column_name}': {e}.")

        # Build the final report
        if not report_lines and not errors:
            return "No outliers detected with the specified methods."

        final_report = "--- Outlier Detection Report ---\n"
        if report_lines:
            final_report += "\n".join(report_lines)
        if errors:
            final_report += "\n\n--- Errors ---\n" + "\n".join(errors)

        return final_report.strip()

    async def handle_outliers(self, outlier_map: Dict[str, str], z_score_threshold: float = 3.0) -> str:
        """
        Handles outliers in numerical columns using either the IQR or Z-score method.

        Args:
            outlier_map: A dictionary mapping column names to the outlier handling strategy ('iqr' or 'z_score').
            z_score_threshold: The threshold for the Z-score method (defaults to 3.0).
        """
        if self.data is None:
            if not self._load_data_from_session():
                return "Error: No data loaded. Please load data first using 'load_data'."

        if not outlier_map:
            return "No outlier map provided; no changes made."

        allowed_strategies = ["iqr", "z_score"]
        successful_removals = []
        errors = []

        for column_name, strategy in outlier_map.items():
            if column_name not in self.data.columns:
                errors.append(f"Error: Column '{column_name}' not found in the dataset.")
                continue

            if strategy not in allowed_strategies:
                errors.append(
                    f"Invalid strategy '{strategy}' for column '{column_name}'. Valid strategies are {allowed_strategies}")
                continue

            try:
                initial_rows = self.data.shape[0]

                if strategy == "iqr":
                    q1 = self.data[column_name].quantile(0.25)
                    q3 = self.data[column_name].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    self.data = self.data.filter(
                        (pl.col(column_name) >= lower_bound) & (pl.col(column_name) <= upper_bound))

                elif strategy == "z_score":
                    mean = self.data[column_name].mean()
                    std = self.data[column_name].std()
                    self.data = self.data.filter(((self.data[column_name] - mean) / std).abs() <= z_score_threshold)

                rows_removed = initial_rows - self.data.shape[0]
                successful_removals.append(
                    f"Successfully removed {rows_removed} outliers from column '{column_name}' using the '{strategy}' method.")

            except Exception as e:
                errors.append(f"Failed to handle outliers in column '{column_name}' with strategy '{strategy}': {e}.")

        self._save_data()

        if len(successful_removals) > 0:
            removal_report = f"""
            Successfully handled outliers in columns: {list(outlier_map.keys())}.

            --- SUCCESSFUL REMOVALS ---
            {'\n'.join(successful_removals)}
            """
            if errors:
                removal_report += f"""
                --- ERRORS ---
                {'\n'.join(errors)}
                """
        else:
            removal_report = f"""
            Could not handle outliers in any of the specified columns.

            --- ERRORS ---
            {'\n'.join(errors)}
            """

        return removal_report.strip()

    async def drop_columns(self, column: Union[str, List[str]]) -> str:
        """
        Drops columns in a DataFrame.

        Args:
            column: Column name or list of column names to drop.
        """
        if self.data is None:
            if not self._load_data_from_session():
                return "Error: No data loaded. Please load data first using 'load_data'."
        try:
            self.data = self.data.drop(column if isinstance(column, list) else [column])
            self._save_data()
            return f"Successfully dropped column(s) '{column}'."
        except Exception as e:
            return f"Error dropping column(s) '{column}': {str(e)}"

    async def drop_rows_with_null(self, column: Optional[Union[str, List[str]]] = None) -> str:
        """
        Drops rows with null values in a DataFrame based on column name and a threshold.
        If no column is specified, all columns will be considered.

        Args:
            column (Optional): Column name or list of column names to subset
        """
        if self.data is None:
            if not self._load_data_from_session():
                return "Error: No data loaded. Please load data first using 'load_data'."

        try:
            self.data = self.data.drop_nulls(subset=column)
            self._save_data()
            return f"Successfully dropped rows with null values."
        except Exception as e:
            return f"Error dropping rows with null values: {e}"

            async def categorize_spending(self, desc_col: str = "description", category_col_name: str = "category",
                                  categories: List[str] = None) -> str:
        """
        Categorizes spending in the transactions DataFrame by extracting information from a specified description column
        and creating a new column with the specified name.

        Args:
            desc_col: The name of the column containing transaction descriptions (defaults to "description").
            category_col_name: The name of the new column to store categories (defaults to "category").
            categories: Optional list of categories for more advanced mapping (currently unused in basic extraction).

        Returns:
            A string message indicating success or error.
        """
        if self.transactions is None:
            if not self._load_data_from_session():
                return "Error: No transactions loaded. Please load data first."
        try:
            possible_desc_cols = [col for col in self.transactions.columns if col.lower() == desc_col.lower()]
            if not possible_desc_cols:
                return f"Error: No column matching '{desc_col}' (case-insensitive) found for descriptions."

            actual_desc_col = possible_desc_cols[0]


            self.transactions = self.transactions.with_columns(
                pl.col(actual_desc_col).str.extract(r'(\w+)').alias(category_col_name)
            )
            self._save_data()
            return f"Spending categorized successfully into column '{category_col_name}' using descriptions from '{actual_desc_col}'."
        except Exception as e:
            return f"Error categorizing spending: {e}"

    async def suggest_savings(self, income_col: str, expense_col: str, goal: float) -> str:
        if self.transactions is None:
            if not self._load_data_from_session():
                return "Error: No transactions loaded. Please load data first."
        try:
            net = self.transactions.select(pl.col(income_col).sum() - pl.col(expense_col).sum()).item()
            suggested_save = min(net * 0.2, goal)
            self.savings_goals = pl.DataFrame(
                {"goal_name": ["Default Goal"], "target_amount": [goal], "suggested_save": [suggested_save]})
            self._save_data()
            return f"Suggest saving {suggested_save} to reach goal of {goal}."
        except Exception as e:
            return f"Error suggesting savings: {e}"

    async def discover_investments(self, query: str = "beginner index funds") -> str:
        return f"Query noted: {query}. Use web_search or browse_page for details. Example: Low-risk ETFs like Vanguard S&P 500. Disclaimer: Not financial advice."

        async def plot_spending_pie(self, category_col: str, amount_col: str = "amount") -> str:
        """
        Generates a pie chart to visualize spending distribution by category using a specified amount column.

        Args:
            category_col: The name of the column containing categories.
            amount_col: The name of the column containing spending amounts (defaults to "amount").

        Returns:
            A string message indicating success or error, including the path to the saved plot if successful.
        """
        if self.transactions is None:
            if not self._load_data_from_session():
                return "Error: No transactions loaded. Please load data first."
        try:
            possible_cat_cols = [col for col in self.transactions.columns if col.lower() == category_col.lower()]
            if not possible_cat_cols:
                return f"Error: No column matching '{category_col}' (case-insensitive) found after categorization."

            actual_cat_col = possible_cat_cols[0]

            possible_amt_cols = [col for col in self.transactions.columns if col.lower() == amount_col.lower()]
            if not possible_amt_cols:
                return f"Error: No column matching '{amount_col}' (case-insensitive) found for amounts."

            actual_amt_col = possible_amt_cols[0]

            sizes = self.transactions.group_by(actual_cat_col).agg(pl.col(actual_amt_col).sum())
            plt.figure()
            plt.pie(sizes[actual_amt_col], labels=sizes[actual_cat_col])
            plt.title("Spending by Category")
            plot_path = os.path.join(self.working_dir, "spending_pie.png")
            plt.savefig(plot_path)
            plt.close()
            return f"Spending pie chart generated and saved at {plot_path}."
        except Exception as e:
            return f"Error generating pie chart: {e}"

session = FinanceBro()


@mcp.tool()
async def wizard_load_data(file_path: str) -> str:
    """
    Load data from a CSV or Excel file into the session for data preprocessing.

    Args:
        file_path: Path of the file to load (must be .csv or .xlsx
    """
    return await session.load_data(file_path)


@mcp.tool()
async def wizard_reset_session() -> str:
    """
    Reset the session for preprocessing by deleting the persisted data and clearing the current state.
    Use this to start fresh without previous modifications.
    """
    return await session.reset_session()


@mcp.tool()
async def wizard_save_to_csv(file_path: str) -> str:
    """
    Saves the current state of the data (after cleaning/transformations) to a new .csv file.

    Args:
        file_path: The desired name for the output file (e.g., 'processed_data.csv').
    """
    return await session.save_to_csv(file_path)


@mcp.tool()
async def wizard_inspect_data(n_rows: int) -> str:
    """
    Performs initial data inspection on the loaded DataFrame to understand the data.
    Includes n rows of the data, DataFrame info, data types of columns, descriptive statistics,
    missing value counts, and duplicate rows.

    Args:
        n_rows: the amount of rows to show (.head())

    Returns:
        Report in the form of a formatted string that summarizes the inspection.
    """
    return await session.inspect_data(n_rows)


@mcp.tool()
async def wizard_plot_distribution(column_name: str, bins: Optional[int] = None) -> str:
    """
    Generates a histogram to visualize the distribution of a selected numerical column

    Args:
        column_name: The name of the column to plot (must be numerical)
        bins (Optional): Number of bins for the histogram (defaults to auto if None)

    """
    return await session.plot_distribution(column_name, bins)


@mcp.tool()
async def wizard_plot_scatter(x_column: str, y_column: str, hue_column: Optional[str] = None) -> str:
    """
    Generates a scatter plot to visualize the relationship between two columns.
    Optionally colors points by a third categorical column (hue).

    Args:
        x_column: The name of the column for the x-axis (numerical or categorical).
        y_column: The name of the column for the y-axis (numerical or categorical).
        hue_column (Optional): The name of a categorical column to color points by (e.g., for grouping).
    """
    return await session.plot_scatter(x_column, y_column, hue_column)


@mcp.tool()
async def wizard_impute_missing_values(imputation_map: Dict[str, str]) -> str:
    """
    Performs imputation on missing values in a column by using a descriptive statistic.
    Takes in a dictionary that maps column names with imputation strategy.
    The column MUST be present in the data frame and the imputation strategy MUST be mean, median, or mode.

    Args:
        imputation_map: Dictionary mapping column names with imputation strategy.

    Returns:
        Report in the form of a formatted string that summarizes the imputation results.
    """

    return await session.impute_missing_values(imputation_map)


@mcp.tool()
async def wizard_convert_to_numeric(column_name: str) -> str:
    """
    Converts values in a column to a numeric type (Float64). If values cannot be converted, set to null.

    Args:
        column_name: The name of the column to convert.
    """

    return await session.convert_to_numeric(column_name)


@mcp.tool()
async def wizard_detect_outliers(outlier_map: Dict[str, str], z_score_threshold: float = 3.0) -> str:
    """
    Detects and reports outliers in numerical columns without removing them.

    Args:
        outlier_map: A dictionary mapping column names to the outlier detection strategy ('iqr' or 'z_score').
                     Example: {"age": "iqr", "income": "z_score"}
        z_score_threshold (Optional): The Z-score threshold for outlier detection (defaults to 3.0).
    """
    return await session.detect_outliers(outlier_map, z_score_threshold)


@mcp.tool()
async def wizard_handle_outliers(outlier_map: Dict[str, str], z_score_threshold: float = 3.0) -> str:
    """
    Handles outliers in numerical columns using either the IQR or Z-score method.

    Args:
        outlier_map: A dictionary mapping column names to the outlier handling strategy ('iqr' or 'z_score').
                     Example: {"age": "iqr", "income": "z_score"}
        z_score_threshold (Optional): The Z-score threshold to use for outlier detection (defaults to 3.0).
    """
    return await session.handle_outliers(outlier_map, z_score_threshold)


@mcp.tool()
async def wizard_drop_columns(column: Union[str, List[str]]) -> str:
    """
    Drops/removes columns in a DataFrame.

    Args:
        column: Column name or list of column names to drop.
    """
    return await session.drop_columns(column)


@mcp.tool()
async def wizard_drop_rows_with_null(column: Optional[Union[str, List[str]]]) -> str:
    """
    Drops rows with null values in a DataFrame based on column name and a threshold.
    If no column is specified, all columns will be considered.

    Args:
        column (Optional): Column name or list of column names to subset
    """

    return await session.drop_rows_with_null(column)


@mcp.tool()
async def wizard_categorize_spending(categories: Optional[List[str]] = None) -> str:
    return await session.categorize_spending(categories)


@mcp.tool()
async def wizard_suggest_savings(income_col: str, expense_col: str, goal: float) -> str:
    return await session.suggest_savings(income_col, expense_col, goal)


@mcp.tool()
async def wizard_discover_investments(query: str) -> str:
    return await session.discover_investments(query)


@mcp.tool()
async def wizard_plot_spending_pie(category_col: str, amount_col: str = "amount") -> str:
    """
    Tool to generate a pie chart for spending by category.

    Args:
        category_col: The name of the category column.
        amount_col: The name of the amount/spending column (defaults to "amount").

    Returns:
        Result message from pie chart generation.
    """
    return await session.plot_spending_pie(category_col, amount_col)


if __name__ == "__main__":
    mcp.run(transport='stdio')
