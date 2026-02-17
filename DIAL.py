import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import io
import json
import datetime
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="DIAL — Decision Intelligence Analytics Lab",
    layout="wide"
)

APP_VERSION = "1.0 Enterprise Demo"

TIERS = ["Starter", "Pro", "Enterprise"]

FEATURE_ACCESS = {
    "EDA": ["Starter","Pro","Enterprise"],
    "ANOMALY": ["Pro","Enterprise"],
    "FORECAST": ["Pro","Enterprise"],
    "SCENARIO": ["Enterprise"],
    "KPI": ["Starter","Pro","Enterprise"],
    "CORRELATION": ["Pro","Enterprise"],
    "NARRATIVE": ["Starter","Pro","Enterprise"]
}

# =========================================================
# SESSION STATE INIT
# =========================================================

if "tier" not in st.session_state:
    st.session_state.tier = "Starter"

if "df" not in st.session_state:
    st.session_state.df = None

# =========================================================
# HEADER
# =========================================================

st.title("Decision Intelligence Analytics Lab")
st.caption(f"Version {APP_VERSION}")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Controls")

tier = st.sidebar.selectbox("License Tier", TIERS)
st.session_state.tier = tier

uploaded = st.sidebar.file_uploader("Upload Dataset", type=["csv","xlsx","json"])

baseline_btn = st.sidebar.button("Generate Baseline Test")

# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data
def load_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    else:
        return pd.read_json(file)

def generate_demo_data():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=200)
    df = pd.DataFrame({
        "date": dates,
        "sales": np.random.normal(200,20,200).cumsum(),
        "cost": np.random.normal(100,15,200).cumsum(),
        "inventory": np.random.randint(50,200,200),
        "region_score": np.random.uniform(0,100,200)
    })
    return df

if uploaded:
    st.session_state.df = load_file(uploaded)

if baseline_btn:
    st.session_state.df = generate_demo_data()

df = st.session_state.df

# =========================================================
# FEATURE ACCESS CHECK
# =========================================================

def allow(feature):
    return st.session_state.tier in FEATURE_ACCESS[feature]

# =========================================================
# DATA PREVIEW
# =========================================================

if df is None:
    st.info("Upload data or generate baseline test")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# =========================================================
# COLUMN DETECTION
# =========================================================

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
date_cols = df.select_dtypes(include=["datetime","datetime64"]).columns.tolist()

# attempt parse date columns
for col in df.columns:
    if "date" in col.lower():
        try:
            df[col] = pd.to_datetime(df[col])
            if col not in date_cols:
                date_cols.append(col)
        except:
            pass

# =========================================================
# EXECUTIVE SUMMARY ENGINE
# =========================================================

def generate_summary(df):
    rows, cols = df.shape
    missing = df.isna().sum().sum()
    msg = f"""
    Dataset contains {rows} rows and {cols} columns.
    Missing values: {missing}.
    Numeric features detected: {len(numeric_cols)}.
    """
    if len(numeric_cols)>0:
        means = df[numeric_cols].mean().sort_values(ascending=False)
        top = means.index[0]
        msg += f" Highest average metric: {top}."
    return msg

st.markdown("### Executive Summary")

if allow("NARRATIVE"):
    st.success(generate_summary(df))
else:
    st.warning("Upgrade tier to unlock executive narrative")

# =========================================================
# TABS
# =========================================================

tabs = st.tabs([
    "Insights",
    "Correlation",
    "Anomalies",
    "Forecast",
    "Scenarios",
    "KPIs"
])

# =========================================================
# INSIGHTS TAB
# =========================================================

with tabs[0]:

    if not allow("EDA"):
        st.warning("Upgrade tier")
    else:
        st.subheader("Distribution Analysis")

        col = st.selectbox("Select metric", numeric_cols)

        fig = px.histogram(df, x=col, nbins=40)
        st.plotly_chart(fig, use_container_width=True)

        st.write("Statistics")
        st.dataframe(df[col].describe())

# =========================================================
# CORRELATION TAB
# =========================================================

with tabs[1]:

    if not allow("CORRELATION"):
        st.warning("Available in Pro+")
    else:
        corr = df[numeric_cols].corr()

        fig = px.imshow(corr, text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# ANOMALY TAB
# =========================================================

with tabs[2]:

    if not allow("ANOMALY"):
        st.warning("Available in Pro+")
    else:
        st.subheader("Anomaly Detection")

        col = st.selectbox("Column", numeric_cols, key="anom")

        X = df[[col]].dropna()
        model = IsolationForest(contamination=0.05)
        preds = model.fit_predict(X)

        df_anom = X.copy()
        df_anom["anomaly"] = preds

        fig = px.scatter(
            df_anom,
            y=col,
            color=df_anom["anomaly"].astype(str)
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# FORECAST TAB
# =========================================================

with tabs[3]:

    if not allow("FORECAST"):
        st.warning("Available in Pro+")
    else:

        if not date_cols:
            st.error("No date column detected")
        else:
            date_col = st.selectbox("Date column", date_cols)
            value_col = st.selectbox("Metric", numeric_cols, key="fc")

            data = df[[date_col,value_col]].rename(
                columns={date_col:"ds", value_col:"y"}
            ).dropna()

            m = Prophet()
            m.fit(data)

            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)

            fig = px.line(forecast, x="ds", y="yhat")
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# SCENARIO TAB
# =========================================================

with tabs[4]:

    if not allow("SCENARIO"):
        st.warning("Enterprise only")
    else:
        st.subheader("What-If Simulator")

        target = st.selectbox("Target metric", numeric_cols)

        factor = st.slider("Impact multiplier", -2.0, 2.0, 0.0, 0.1)

        sim = df[target] * (1 + factor)

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df[target], name="Original"))
        fig.add_trace(go.Scatter(y=sim, name="Simulated"))
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# KPI TAB
# =========================================================

with tabs[5]:

    if not allow("KPI"):
        st.warning("Upgrade tier")
    else:
        st.subheader("Auto KPI Builder")

        metric = st.selectbox("Metric", numeric_cols, key="kpi")

        col1,col2,col3 = st.columns(3)

        col1.metric("Mean", round(df[metric].mean(),2))
        col2.metric("Max", round(df[metric].max(),2))
        col3.metric("Min", round(df[metric].min(),2))

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.caption("DIAL Platform — Decision Intelligence Engine")
