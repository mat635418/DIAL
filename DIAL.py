import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(page_title="DIAL Platform", layout="wide")

# =========================================
# LOGIN SYSTEM (reads from secrets)
# =========================================
def login():
    if "logged" not in st.session_state:
        st.session_state.logged = False

    if st.session_state.logged:
        return True

    st.title("Secure Login")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if (
            user == st.secrets["auth"]["username"]
            and pwd == st.secrets["auth"]["password"]
        ):
            st.session_state.logged = True
            st.rerun()
        else:
            st.error("Invalid credentials")

    return False


if not login():
    st.stop()


# =========================================
# SIDEBAR CONFIG PANEL
# =========================================
with st.sidebar.expander("Configuration", expanded=False):
    volatility_threshold = st.slider("Volatility Alert Threshold", 0.0, 2.0, 0.8)
    anomaly_sensitivity = st.slider("Anomaly Sensitivity", 1.0, 5.0, 2.5)
    forecast_periods = st.slider("Forecast Periods", 3, 30, 10)
    decision_threshold = st.slider("Decision Sensitivity %", 1, 50, 15)

# =========================================
# DATA UPLOAD SECTION
# =========================================
st.title("MEIO Intelligence Suite")

st.markdown("### Dataset Upload")

uploaded = st.file_uploader(
    "Upload dataset",
    type=["csv", "xlsx"],
    help="Upload CSV or Excel file with numeric business metrics. Columns should represent KPIs and rows should represent time or observations.",
)

if st.button("Generate Demo Dataset"):
    rows = 200
    df = pd.DataFrame({
        "revenue": np.random.normal(500, 100, rows).cumsum(),
        "cost": np.random.normal(300, 80, rows).cumsum(),
        "inventory": np.random.normal(200, 60, rows).cumsum(),
        "demand": np.random.normal(400, 120, rows).cumsum(),
        "lead_time": np.abs(np.random.normal(10, 3, rows))
    })
else:
    if uploaded:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    else:
        st.info("Upload dataset or generate demo data")
        st.stop()

# =========================================
# DATA SUMMARY
# =========================================
numeric_df = df.select_dtypes(include="number")

st.success(
    f"Rows {df.shape[0]} | Columns {df.shape[1]} | Numeric {numeric_df.shape[1]} | Missing {df.isna().sum().sum()}"
)

# =========================================
# AUTONOMOUS ALERT ENGINE
# =========================================
st.markdown("### Autonomous Alerts")

alerts = []

for col in numeric_df.columns:
    series = numeric_df[col]
    vol = series.std() / (abs(series.mean()) + 1e-5)

    if vol > volatility_threshold:
        alerts.append(f"Risk: {col} extremely volatile")

    if series.isna().sum() > 0:
        alerts.append(f"Data issue: {col} contains missing values")

if alerts:
    for a in alerts:
        st.error(a)
else:
    st.success("No critical alerts")

# =========================================
# TABS
# =========================================
tabs = st.tabs([
    "Insights",
    "KPIs",
    "Correlation",
    "Anomalies",
    "Forecast",
    "Drivers",
    "Scenario",
    "Risk",
    "Decision",
    "AI Copilot",
    "Export"
])

# =========================================
# INSIGHTS
# =========================================
with tabs[0]:
    st.subheader("Quick Insights")
    st.write(numeric_df.describe())

# =========================================
# KPI
# =========================================
with tabs[1]:
    st.subheader("KPI Monitor")
    col = st.selectbox("Metric ?", numeric_df.columns)
    st.line_chart(numeric_df[col])

# =========================================
# CORRELATION
# =========================================
with tabs[2]:
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots()
    cax = ax.matshow(numeric_df.corr())
    fig.colorbar(cax)
    st.pyplot(fig)

# =========================================
# ANOMALIES
# =========================================
with tabs[3]:
    st.subheader("Anomaly Detection")
    col = st.selectbox("Column ?", numeric_df.columns, key="anom")

    series = numeric_df[col]
    z = (series - series.mean()) / series.std()
    anomalies = series[abs(z) > anomaly_sensitivity]

    st.write("Detected anomalies:", len(anomalies))
    st.dataframe(anomalies)

# =========================================
# FORECAST
# =========================================
with tabs[4]:
    st.subheader("Forecast")

    col = st.selectbox("Forecast Metric ?", numeric_df.columns, key="forecast")
    series = numeric_df[col]

    trend = np.polyfit(range(len(series)), series, 1)[0]
    future = [series.iloc[-1] + trend * i for i in range(forecast_periods)]

    st.line_chart(list(series) + future)

# =========================================
# DRIVERS
# =========================================
with tabs[5]:
    st.subheader("Driver Analysis")
    target = st.selectbox("Target ?", numeric_df.columns)

    corr = numeric_df.corr()[target].drop(target).sort_values(key=abs, ascending=False)
    st.bar_chart(corr)

# =========================================
# SCENARIO
# =========================================
with tabs[6]:
    st.subheader("Scenario Simulation")

    col = st.selectbox("Scenario Metric ?", numeric_df.columns, key="scen")
    change = st.slider("Change %", -50, 50, 10)

    new = numeric_df[col].iloc[-1] * (1 + change/100)

    st.metric("Projected Value", f"{new:.2f}")

# =========================================
# RISK
# =========================================
with tabs[7]:
    st.subheader("Risk Scoring")

    risks = {}
    for col in numeric_df.columns:
        risks[col] = numeric_df[col].std()

    risk_df = pd.DataFrame.from_dict(risks, orient="index", columns=["Risk"])
    st.bar_chart(risk_df)

# =========================================
# DECISION ENGINE (PRO VERSION)
# =========================================
with tabs[8]:
    st.subheader("Strategic Decision Engine")

    metric = st.selectbox("Metric ?", numeric_df.columns, key="decision")

    series = numeric_df[metric]

    change_pct = ((series.iloc[-1] - series.iloc[0]) / (abs(series.iloc[0]) + 1e-5)) * 100

    st.metric("Trend %", f"{change_pct:.2f}%")

    if change_pct > decision_threshold:
        st.success("Strong growth — Increase investment")

    elif 5 < change_pct <= decision_threshold:
        st.info("Moderate growth — Maintain strategy")

    elif -5 <= change_pct <= 5:
        st.warning("Flat performance — Optimize operations")

    else:
        st.error("Decline detected — Immediate action required")

# =========================================
# AI COPILOT
# =========================================
with tabs[9]:
    st.subheader("AI Copilot")

    question = st.text_input("Ask a question about your data")

    if question:
        st.info("AI Insight")

        desc = numeric_df.describe().to_string()

        st.write(
            f"""
Based on dataset statistics:

{desc}

Suggested interpretation:
- Highest variability column likely risk driver
- Highest mean column likely dominant KPI
- Lowest values could signal inefficiencies

(User question: {question})
"""
        )

# =========================================
# EXPORT
# =========================================
with tabs[10]:
    st.subheader("Export Data")

    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        file_name="export.csv"
    )
