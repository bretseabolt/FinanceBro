## FinanceBro

A smart budgeting and AI-driven financial assistant that helps users manage spending, build saving habits, and discover beginner-friendly investment opportunities

## Installation

### Prerequisites
- Python 3.12+
- Git

### Steps
1. Clone the repository:
   ```
   git clone https://github.com/bretseabolt/FinanceBro.git
   cd FinanceBro
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

   *Note*: The project uses libraries like `streamlit`, `polars`, `scikit-learn`, `langgraph`, `langchain`, `joblib`, and `dotenv`. Ensure `MCP_FILESYSTEM_DIR` is set in `.env` for data storage (defaults to `./data`).

4. Set up environment variables:
   Create a `.env` file in the root directory:
   ```
   MCP_FILESYSTEM_DIR=./data
   GOOGLE_API_KEY=your_google_api_key  # For Gemini LLM in graph.py
   ```

   ## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. **Main Page (FinanceBro)**:
   - Upload a CSV or Excel file via the file uploader.
   - Interact with the AI agent via chat input
   - Use commands like "Reset session" to clear state.

3. **Data Viewer Page**:
   - View current DataFrames

4. **Visualization Page**:
   - Vuew visualizations on your data
     
The MCP server (`finance_bro.py`) runs in the background for tool execution. Sessions persist via Parquet files and joblib.
