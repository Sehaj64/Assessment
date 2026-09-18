"""
Streamlit UI for Support Ticket AI Intelligence System.
Enables interactive natural language querying, anomaly exploration,
and operational metrics visualization.
Run with: streamlit run app_streamlit.py
"""

import streamlit as st
import pandas as pd
from analytics.anomaly_detector import AnomalyDetector
from core.database import DatabaseManager
from services.query_engine import QueryEngine

st.set_page_config(
    page_title="TicketPulse AI - Streamlit Dashboard",
    page_icon="🎫",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .stMetric {
        background-color: #111827;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 14px 18px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Core Services (cached)
@st.cache_resource
def get_services():
    db = DatabaseManager.get_instance()
    detector = AnomalyDetector(db)
    engine = QueryEngine(db)
    return db, detector, engine

db, detector, engine = get_services()
stats = db.get_summary_stats()
provider_info = engine.llm.get_provider_info()

# Sidebar Settings & Diagnostics
st.sidebar.title("⚙️ System Control")
st.sidebar.markdown(f"**Status:** 🟢 `Operational`")
st.sidebar.markdown(f"**Active Engine:** `{provider_info['provider'].upper()}`")
st.sidebar.caption(f"Model: `{provider_info['model']}`")

st.sidebar.divider()
st.sidebar.markdown("### 🔑 API Key (Optional)")
custom_key = st.sidebar.text_input("Gemini API Key", type="password", placeholder="Enter key for live cloud LLM", help="If deploying on Streamlit Cloud, enter your free Gemini API key here or leave blank to use the offline semantic engine.")
if custom_key and custom_key != engine.llm.gemini_key:
    from services.llm_client import LLMClient
    engine.llm = LLMClient(override_gemini_key=custom_key)
    st.sidebar.success("Custom Gemini Key Activated!")

st.sidebar.divider()
st.sidebar.markdown("### 🔗 Quick Links")
st.sidebar.markdown("- [GitHub Repository](https://github.com/Sehaj64/Assessment)")
st.sidebar.markdown("- [FastAPI Backend Docs](http://localhost:8000/docs)")

# Header
st.title("🎫 TicketPulse AI Intelligence System")
st.caption(f"DOTMappers Technical Assessment | Active LLM: **{engine.llm.active_provider.upper()}** ({engine.llm.active_model}) | Zero-Cost Architecture")

# Top KPI Metric Cards
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Tickets", f"{stats['total_tickets']:,}")
col2.metric("Unresolved Tickets", f"{stats['unresolved_count']:,}", delta="-Open & Escalated", delta_color="inverse")
col3.metric("Avg Resolution Time", f"{stats['avg_resolution_time_hrs']} hrs")
col4.metric("Avg Customer Rating", f"{stats['avg_customer_rating']} / 5")
anomalies_data = detector.detect_all()
col5.metric("Flagged Anomalies", f"{anomalies_data['total_anomalies_flagged']:,}", delta="SLA & ML Outliers", delta_color="inverse")

st.divider()

# Tabs
tab_query, tab_anomalies, tab_tickets = st.tabs(["🔍 Natural Language Query", "🚨 Anomaly Radar", "📋 Ticket Explorer"])

# TAB 1: NATURAL LANGUAGE QUERY
with tab_query:
    st.subheader("Ask Natural Language Questions")
    st.write("Translates plain English queries to SQLite, executes against the dataset, and synthesizes executive answers.")

    sample_questions = [
        "How many tickets are currently open?",
        "Which agent resolved the most tickets this month?",
        "Show me all Critical tickets not resolved within 12 hours.",
        "What is the average customer rating for Technical category tickets?",
        "Are there any anomalies in resolution times this week?",
        "How many critical tickets are unresolved?",
        "Which agent has the lowest average customer rating?"
    ]

    selected_sample = st.selectbox("Or choose a sample question:", [""] + sample_questions)
    default_text = selected_sample if selected_sample else ""
    user_query = st.text_input("Enter your natural language question:", value=default_text, placeholder="e.g. How many critical tickets are unresolved?")

    if st.button("Execute Query", type="primary"):
        if user_query.strip():
            with st.spinner("Processing query via Text-to-SQL and LLM synthesis..."):
                result = engine.ask(user_query)

            st.success("Query Executed Successfully!")
            
            # Answer Box
            st.markdown("### 💡 Executive AI Summary")
            st.markdown(result["answer"])

            st.markdown("---")
            col_meta1, col_meta2, col_meta3 = st.columns(3)
            col_meta1.caption(f"**Provider:** `{result['llm_provider']}`")
            col_meta2.caption(f"**Latency:** `{result['execution_time_ms']} ms`")
            col_meta3.caption(f"**Rows Returned:** `{result['row_count']}`")

            # SQL Accordion
            with st.expander("Inspect Generated SQLite Query", expanded=True):
                st.code(result["sql"], language="sql")

            # Data Table
            if result["data"]:
                st.markdown("### 📊 Retrieved Data")
                df_res = pd.DataFrame(result["data"])
                st.dataframe(df_res)
        else:
            st.warning("Please enter a question.")

# TAB 2: ANOMALY RADAR
with tab_anomalies:
    st.subheader("Support Ticket Anomaly Radar")
    st.write("Multi-layered anomaly detection combining SLA rules (e.g. unresolved critical tickets > 24h), statistical IQR/Z-score outliers, and Scikit-Learn Isolation Forest.")

    sev_filter = st.radio("Filter by Severity:", ["ALL", "CRITICAL", "HIGH", "MEDIUM"], horizontal=True)

    anomalies_list = anomalies_data["anomalies"]
    if sev_filter != "ALL":
        anomalies_list = [a for a in anomalies_list if a["severity"] == sev_filter]

    st.write(f"Showing **{len(anomalies_list)}** anomalies:")

    for a in anomalies_list[:50]:
        sev_color = "red" if a["severity"] == "CRITICAL" else ("orange" if a["severity"] == "HIGH" else "blue")
        with st.container():
            st.markdown(f"""
            **Ticket `{a['ticket_id']}`** | :{sev_color}[{a['severity']}] | Category: **{a['category']}** | Priority: **{a['priority']}** | Agent: **{a['agent_id']}** | Type: `{a['anomaly_type']}`
            - **Reason**: {a['reason']}
            - **Metric**: `{a['metric_name']}` = `{a['metric_value']}` (Threshold: `{a['threshold']}`)
            - **Recommended Action**: *{a['recommended_action']}*
            """)
            st.divider()

# TAB 3: TICKET EXPLORER
with tab_tickets:
    st.subheader("Complete Ticket Dataset Explorer")
    conn = db.get_connection()
    raw_df = pd.read_sql_query("SELECT * FROM tickets", conn)

    col_f1, col_f2, col_f3 = st.columns(3)
    cat_select = col_f1.multiselect("Filter Category", options=raw_df["category"].unique())
    prio_select = col_f2.multiselect("Filter Priority", options=raw_df["priority"].unique())
    stat_select = col_f3.multiselect("Filter Status", options=raw_df["status"].unique())

    filtered_df = raw_df.copy()
    if cat_select:
        filtered_df = filtered_df[filtered_df["category"].isin(cat_select)]
    if prio_select:
        filtered_df = filtered_df[filtered_df["priority"].isin(prio_select)]
    if stat_select:
        filtered_df = filtered_df[filtered_df["status"].isin(stat_select)]

    st.write(f"Showing {len(filtered_df)} of {len(raw_df)} tickets:")
    st.dataframe(filtered_df)
