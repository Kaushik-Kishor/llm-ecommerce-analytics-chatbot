import os
import textwrap
import json

import duckdb
import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings


# ---------- Paths & Globals ----------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PROCESSED = os.path.join(BASE_DIR, "..", "data_processed")
CORE_CSV_PATH = os.path.join(DATA_PROCESSED, "olist_core_orders.csv")
FAISS_PATH = os.path.join(DATA_PROCESSED, "faiss_olist_reviews")

_df_cache = None
_vectorstore_cache = None
_embeddings_cache = None
_llm_cache = None


# ---------- Embeddings Wrapper ----------

class LocalEmbeddings(Embeddings):
    """Wrapper so LangChain can use a local SentenceTransformer model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        global _embeddings_cache
        if _embeddings_cache is None:
            _embeddings_cache = SentenceTransformer(model_name)
        self.model = _embeddings_cache

    def embed_documents(self, texts):
        return self.model.encode(
            texts,
            show_progress_bar=False,
            device="cpu",
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            [text],
            device="cpu",
        )[0].tolist()


# ---------- Lazy loaders ----------

def get_df() -> pd.DataFrame:
    """Load the main analytics table (cached)."""
    global _df_cache
    if _df_cache is None:
        print("Loading core orders dataframe...")
        _df_cache = pd.read_csv(
            CORE_CSV_PATH,
            parse_dates=["order_purchase_timestamp"],
        )
    return _df_cache


def get_vectorstore() -> FAISS:
    """Load FAISS index (cached)."""
    global _vectorstore_cache
    if _vectorstore_cache is None:
        print("Loading FAISS index...")
        embedding = LocalEmbeddings("all-MiniLM-L6-v2")
        _vectorstore_cache = FAISS.load_local(
            FAISS_PATH,
            embedding,
            allow_dangerous_deserialization=True,
        )
    return _vectorstore_cache


def get_llm():
    """Get the local LLM (Mistral via Ollama)."""
    global _llm_cache
    if _llm_cache is None:
        print("Initializing Ollama LLM (mistral)...")
        _llm_cache = Ollama(model="mistral")
    return _llm_cache


# ---------- KPI & Trend Helpers ----------

def basic_kpi_summary(df: pd.DataFrame) -> str:
    total_orders = df["order_id"].nunique()
    total_customers = df["customer_unique_id"].nunique()
    total_revenue = df["payment_value_sum"].sum()

    avg_review = df["review_score"].mean()
    late_ratio = df["is_late"].mean() * 100

    cat_rev = (
        df.groupby("product_category_clean")["payment_value_sum"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )

    cat_lines = "\n".join(
        [f"- {idx}: {val:,.2f} BRL" for idx, val in cat_rev.items()]
    )

    summary = f"""
    High-level KPIs:
    - Total unique orders: {total_orders:,}
    - Total unique customers: {total_customers:,}
    - Total revenue (all time): {total_revenue:,.2f} BRL
    - Average review score: {avg_review:.2f} / 5
    - % of orders delivered late: {late_ratio:.2f}%

    Top 5 categories by revenue:
    {cat_lines}
    """
    return textwrap.dedent(summary).strip()


def compute_kpi_numbers(df: pd.DataFrame) -> dict:
    """Return raw numbers for Streamlit KPI cards."""
    return {
        "total_orders": int(df["order_id"].nunique()),
        "total_customers": int(df["customer_unique_id"].nunique()),
        "total_revenue": float(df["payment_value_sum"].sum()),
        "avg_review": float(df["review_score"].mean()),
        "late_ratio": float(df["is_late"].mean() * 100.0),
    }


def time_trend_summary(df: pd.DataFrame) -> str:
    df_month = df.copy()
    df_month["order_month"] = df_month["order_purchase_timestamp"].dt.to_period("M")
    monthly_rev = (
        df_month.groupby("order_month")["payment_value_sum"]
        .sum()
        .sort_index()
    )

    last_vals = monthly_rev.tail(6)
    lines = "\n".join(
        [f"- {str(idx)}: {val:,.2f} BRL" for idx, val in last_vals.items()]
    )

    direction = "increased"
    if len(last_vals) >= 2 and last_vals.iloc[-1] < last_vals.iloc[0]:
        direction = "decreased"

    summary = f"""
    Recent revenue trend (last {len(last_vals)} months):

    {lines}

    Overall, revenue has {direction} over this period (simple start vs end comparison).
    """
    return textwrap.dedent(summary).strip()


# ---------- System Context ----------

def build_system_context() -> str:
    return textwrap.dedent(
        """
        You are an e-commerce analytics assistant.
        You are working with a Brazilian marketplace dataset (Olist) where each row is an order item
        joined with customer, product, seller, payment and review information.

        Important columns you may reference in your reasoning:
        - order_purchase_timestamp: when the order was placed
        - payment_value_sum: total payment value for the order (revenue)
        - payment_installments_max: number of installments
        - main_payment_type: payment method (credit_card, boleto, etc.)
        - product_category_clean: cleaned/English product category name
        - review_score: 1–5 star rating
        - review_comment_message: free-text review from the customer
        - delivery_days: actual delivery time in days
        - estimated_delivery_days: estimated delivery time in days
        - is_late: 1 if delivery_days > estimated_delivery_days else 0
        - customer_state, customer_city: customer location
        - seller_state, seller_city: seller location

        Combine:
        - quantitative insights (aggregations over the dataframe)
        - qualitative insights from retrieved reviews (FAISS search)
        and optionally mention what charts / tables might be useful.
        """
    ).strip()


# ---------- Core QA with RAG + KPIs ----------

def format_history_for_prompt(chat_history):
    """Turn chat history list of (role, text) into a brief string."""
    if not chat_history:
        return ""
    # last 4 exchanges max
    last = chat_history[-8:]
    lines = []
    for role, msg in last:
        lines.append(f"{role}: {msg}")
    return "\n".join(lines)


def answer_question(query: str, k: int = 5, chat_history=None) -> str:
    """
    Main entry point:
    - retrieves relevant reviews via FAISS
    - computes basic KPIs & trends
    - sends everything + short history to the local LLM for a final answer
    """
    df = get_df()
    db = get_vectorstore()
    llm = get_llm()

    retrieved_docs = db.similarity_search(query, k=k)
    reviews_text = "\n\n".join([d.page_content for d in retrieved_docs])

    kpi_text = basic_kpi_summary(df)
    trend_text = time_trend_summary(df)
    system_ctx = build_system_context()
    history_text = format_history_for_prompt(chat_history)

    prompt = f"""
    {system_ctx}

    Conversation so far (if any):
    {history_text}

    Below are some high-level metrics about the full dataset:
    {kpi_text}

    And a brief recent revenue trend:
    {trend_text}

    I also retrieved a few example customer reviews related to the question:
    {reviews_text}

    User question:
    {query}

    Task:
    - Use the metrics, trends, and reviews above to answer the question.
    - Provide clear, structured insights (bullet points are fine).
    - Stay grounded in the provided data and reviews.
    """

    prompt = textwrap.dedent(prompt).strip()
    response = llm.invoke(prompt)
    return response


# ---------- Natural-language → SQL ----------

def _schema_text(df: pd.DataFrame) -> str:
    return "\n".join([f"{col}: {str(dtype)}" for col, dtype in zip(df.columns, df.dtypes)])


def nl_to_sql(question: str) -> str:
    """
    Use the LLM to convert a natural-language analytics question
    into a DuckDB SQL query over table 'olist'.
    """
    df = get_df()
    llm = get_llm()

    schema = _schema_text(df)

    prompt = f"""
    You are a senior data analyst using DuckDB SQL.

    You have a single table called olist with the following columns and dtypes:
    {schema}

    Write ONE DuckDB-compatible SQL query that answers the following question:

    "{question}"

    Rules:
    - Use table name: olist
    - Prefer aggregations like SUM, AVG, COUNT when appropriate.
    - If the question asks for trends over time, group by a time period (e.g. month).
    - Do NOT use backticks.
    - Do NOT explain the query.
    - Return ONLY the SQL query, nothing else.
    """

    sql = llm.invoke(textwrap.dedent(prompt).strip())
    
    sql = str(sql).strip()
    if sql.startswith("```"):
        sql = sql.strip("`")
        
        sql = "\n".join(line for line in sql.splitlines() if not line.strip().lower().startswith("sql"))
    return sql.strip()


def run_sql_over_df(sql: str, limit: int = 200) -> tuple[pd.DataFrame | None, str | None]:
    df = get_df()
    try:
        con = duckdb.connect()
        con.register("olist", df)
        
        final_sql = sql
        if "limit" not in sql.lower():
            final_sql = sql.rstrip(";") + f"\nLIMIT {limit};"
        result = con.execute(final_sql).df()
        con.close()
        return result, None
    except Exception as e:
        return None, str(e)


def answer_sql_question(question: str, limit: int = 200) -> dict:
    """
    High-level helper:
    - generate SQL from natural language
    - run SQL
    - optionally get a short interpretation from the LLM
    """
    sql = nl_to_sql(question)
    df_result, err = run_sql_over_df(sql, limit=limit)

    explanation = None
    if df_result is not None and err is None:
        llm = get_llm()
        preview = df_result.head(10).to_markdown(index=False)
        explain_prompt = f"""
        You are a data analyst.

        I asked the question:
        "{question}"

        You wrote and ran the following SQL on an e-commerce table:

        ```sql
        {sql}
        ```

        Here is the first few rows of the result table:

        {preview}

        Briefly explain what this result shows in 3–6 bullet points.
        """
        explanation = llm.invoke(textwrap.dedent(explain_prompt).strip())

    return {
        "sql": sql,
        "result": df_result,
        "error": err,
        "explanation": explanation,
    }


# ---------- Natural-language → Chart Plan ----------

def plan_chart(request: str) -> dict | str:
    """
    Ask the LLM to plan a chart in a constrained JSON format.
    We keep it simple: bar / line / hist.
    """
    df = get_df()
    llm = get_llm()
    schema = _schema_text(df)

    prompt = f"""
    You are a data visualization planner.

    You have a pandas DataFrame called df with the following columns and dtypes:
    {schema}

    The user wants this chart:
    "{request}"

    Choose ONE of these chart types: "bar", "line", "hist".

    Respond with a JSON object ONLY, no extra text, with keys:
    - "chart_type": "bar" | "line" | "hist"
    - "x": column name for x-axis (string)
    - "y": column name for y-axis or aggregation target (string or null)
    - "agg": aggregation to apply for y if needed: "sum", "mean", "count", or null
    - "note": short note to describe what you planned.

    Examples:
    {{"chart_type": "bar", "x": "product_category_clean", "y": "payment_value_sum", "agg": "sum", "note": "Revenue by category"}}
    {{"chart_type": "hist", "x": "review_score", "y": null, "agg": null, "note": "Distribution of review scores"}}

    Return ONLY valid JSON.
    """

    raw = llm.invoke(textwrap.dedent(prompt).strip())
    text = str(raw).strip()

    if text.startswith("```"):
        
        text = text.strip("`")
        lines = [ln for ln in text.splitlines() if not ln.strip().lower().startswith("json")]
        text = "\n".join(lines)

    try:
        plan = json.loads(text)
        return plan
    except Exception as e:
        return f"Failed to parse chart plan JSON. Raw response was:\n{text}\n\nError: {e}"


# ---------- Simple CLI for debugging ----------

if __name__ == "__main__":
    df = get_df()
    print("DF shape:", df.shape)
    print("Example KPIs:", compute_kpi_numbers(df))
    while True:
        q = input("\nAsk a question (or 'sql:' prefix for SQL mode, 'chart:' for chart plan, 'exit'):\n> ")
        if q.lower().strip() in {"exit", "quit"}:
            break
        if q.lower().startswith("sql:"):
            out = answer_sql_question(q[4:].strip())
            print("SQL:\n", out["sql"])
            print("Error:", out["error"])
            if out["result"] is not None:
                print(out["result"].head())
            print("Explanation:\n", out["explanation"])
        elif q.lower().startswith("chart:"):
            plan = plan_chart(q[6:].strip())
            print("Plan:", plan)
        else:
            ans = answer_question(q)
            print("\nAssistant:\n", ans)
