import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from chatbot_engine import (
    answer_question,
    get_df,
    compute_kpi_numbers,
    answer_sql_question,
    plan_chart,
)


st.set_page_config(
    page_title="E-commerce LLM Analytics Assistant",
    layout="wide"
)

st.title("🛒 E-commerce LLM Analytics Assistant")
st.caption(
    "Local LLM + FAISS + DuckDB • Ask questions, run analytics, and generate charts over the Olist e-commerce dataset."
)

df = get_df()
kpis = compute_kpi_numbers(df)

# ---------- KPI CARDS ----------
st.markdown("### 📊 Key Metrics")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Orders", f"{kpis['total_orders']:,}")
with c2:
    st.metric("Total Customers", f"{kpis['total_customers']:,}")
with c3:
    st.metric("Total Revenue (BRL)", f"{kpis['total_revenue']:,.2f}")
with c4:
    st.metric("Avg Review Score", f"{kpis['avg_review']:.2f}")

c5, _ = st.columns([1, 3])
with c5:
    st.metric("% Late Deliveries", f"{kpis['late_ratio']:.2f}%")

st.divider()

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Options")

k_value = st.sidebar.slider("Number of reviews to retrieve (k)", 1, 10, 5)

if st.sidebar.checkbox("Show raw sample data"):
    st.sidebar.write(df.sample(5))

if st.sidebar.button("Show Review Score Distribution"):
    fig, ax = plt.subplots(figsize=(5, 3))
    df["review_score"].hist(bins=5, ax=ax)
    ax.set_title("Distribution of Review Scores")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    st.pyplot(fig)

if st.sidebar.button("Clear Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()


# ---------- CHAT MEMORY SETUP ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  


# ---------- MAIN LAYOUT: Chat + Tools ----------
chat_col, tools_col = st.columns([2, 1])

## ----- Chat column -----
with chat_col:

    
    def send_query():
        user_message = st.session_state.chat_input.strip()
        if user_message:
            
            st.session_state.chat_history.append(("User", user_message))

            
            answer = answer_question(
                user_message,
                k=k_value,
                chat_history=st.session_state.chat_history,
            )

            
            st.session_state.chat_history.append(("Assistant", str(answer)))

        
        st.session_state.chat_input = ""

    st.subheader("💬 Chat with the Analytics Assistant")

    
    st.text_input(
        "Ask a question about customers, revenue, reviews, delivery, etc.",
        key="chat_input",
        on_change=send_query,
    )

    
    for role, msg in st.session_state.chat_history:
        if role == "User":
            st.markdown(f"**🧑 User:** {msg}")
        else:
            st.markdown(f"**🤖 Assistant:** {msg}")

# ----- Tools column -----
with tools_col:
    st.subheader("🧪 Analytics Lab")

    st.markdown("#### 1) Natural-language → Chart")

    chart_request = st.text_area(
        "Describe the chart you want (example: 'Revenue by product category as bar chart', "
        "'Trend of average review score over time', 'Histogram of delivery_days').",
        height=80,
        key="chart_request",
    )

    if st.button("Generate Chart Plan & Plot"):
        if chart_request.strip():
            plan = plan_chart(chart_request.strip())
            if isinstance(plan, str):
                st.error(plan)
            else:
                st.markdown("**Chart Plan (from LLM):**")
                st.json(plan)

                chart_type = plan.get("chart_type")
                x_col = plan.get("x")
                y_col = plan.get("y")
                agg = plan.get("agg")

                if chart_type not in {"bar", "line", "hist"}:
                    st.error(f"Unsupported chart_type: {chart_type}")
                elif x_col not in df.columns:
                    st.error(f"x column '{x_col}' not found in dataframe.")
                elif y_col is not None and y_col not in df.columns:
                    st.error(f"y column '{y_col}' not found in dataframe.")
                else:
                    fig, ax = plt.subplots(figsize=(5, 3))

                    if chart_type == "hist":
                        
                        try:
                            df[x_col].hist(ax=ax, bins=20)
                        except Exception as e:
                            st.error(f"Failed to build histogram: {e}")
                        ax.set_title(plan.get("note", "Histogram"))
                        ax.set_xlabel(x_col)
                        ax.set_ylabel("Count")

                    else:
                        
                        data = df.copy()
                        if y_col is None:
                            
                            grouped = data.groupby(x_col).size()
                            y_label = "count"
                        else:
                            if agg == "sum":
                                grouped = data.groupby(x_col)[y_col].sum()
                            elif agg == "mean":
                                grouped = data.groupby(x_col)[y_col].mean()
                            elif agg == "count":
                                grouped = data.groupby(x_col)[y_col].count()
                            else:
                                
                                grouped = data.groupby(x_col)[y_col].sum()
                            y_label = f"{agg or 'sum'}({y_col})"

                        grouped = grouped.sort_values(ascending=False).head(20)

                        if chart_type == "bar":
                            grouped.plot(kind="bar", ax=ax)
                        elif chart_type == "line":
                            grouped.plot(kind="line", marker="o", ax=ax)

                        ax.set_title(plan.get("note", "Chart"))
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y_label)
                        plt.xticks(rotation=45, ha="right")

                    st.pyplot(fig)

    st.markdown("---")
    st.markdown("#### 2) Natural-language → SQL Analytics")

    sql_question = st.text_area(
        "Ask a question that should be answered with a table/query "
        "(example: 'Total revenue by month', 'Top 10 categories by revenue', "
        "'Average review_score by product_category_clean').",
        height=80,
        key="sql_question",
    )

    if st.button("Run SQL Analysis"):
        if sql_question.strip():
            with st.spinner("Thinking and running SQL..."):
                out = answer_sql_question(sql_question.strip())

            st.markdown("**Generated SQL:**")
            st.code(out["sql"], language="sql")

            if out["error"]:
                st.error(f"SQL error: {out['error']}")
            elif out["result"] is not None:
                st.markdown("**Result (preview):**")
                st.dataframe(out["result"].head(100))

                if out["explanation"]:
                    st.markdown("**Explanation:**")
                    st.write(out["explanation"])
            else:
                st.warning("No result returned.")
