import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListItem, ListFlowable
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import datetime
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(page_title="DIAL Platform", layout="wide")

TIERS = ["Starter","Pro","Enterprise"]

ACCESS = {
    "eda":["Starter","Pro","Enterprise"],
    "kpi":["Starter","Pro","Enterprise"],
    "summary":["Starter","Pro","Enterprise"],

    "corr":["Pro","Enterprise"],
    "anomaly":["Pro","Enterprise"],
    "forecast":["Pro","Enterprise"],
    "drivers":["Pro","Enterprise"],
    "multidata":["Pro","Enterprise"],

    "scenario":["Enterprise"],
    "risk":["Enterprise"],
    "pdf":["Enterprise"],
    "decision":["Enterprise"]
}

if "tier" not in st.session_state:
    st.session_state.tier="Starter"

if "datasets" not in st.session_state:
    st.session_state.datasets={}

# =========================================================
# UTILS
# =========================================================

def allow(f):
    return st.session_state.tier in ACCESS[f]

def load_file(file):
    if file.name.endswith("csv"):
        return pd.read_csv(file)
    if file.name.endswith("xlsx"):
        return pd.read_excel(file)
    return pd.read_json(file)

def detect_columns(df):
    nums=df.select_dtypes(include=np.number).columns.tolist()
    dates=[]
    for c in df.columns:
        if "date" in c.lower():
            try:
                df[c]=pd.to_datetime(df[c])
                dates.append(c)
            except: pass
    return nums,dates

# =========================================================
# HEADER
# =========================================================

st.title("DIAL — Decision Intelligence Analytics Lab")

st.sidebar.header("Control Center")
tier=st.sidebar.selectbox("License Tier",TIERS)
st.session_state.tier=tier

uploaded=st.sidebar.file_uploader("Upload Dataset",accept_multiple_files=True)

# =========================================================
# DATA LOAD
# =========================================================

if uploaded:
    for f in uploaded:
        st.session_state.datasets[f.name]=load_file(f)

if not st.session_state.datasets:
    st.info("Upload dataset(s)")
    st.stop()

dataset_name=st.selectbox("Select Dataset",list(st.session_state.datasets.keys()))
df=st.session_state.datasets[dataset_name]

num_cols,date_cols=detect_columns(df)

# =========================================================
# EXECUTIVE SUMMARY
# =========================================================

if allow("summary"):
    st.subheader("Executive Summary")
    st.success(
        f"Rows: {df.shape[0]} | Columns: {df.shape[1]} | Numeric: {len(num_cols)} | Missing: {df.isna().sum().sum()}"
    )

# =========================================================
# TABS
# =========================================================

tabs=st.tabs([
"Insights","KPIs","Correlation","Anomalies",
"Forecast","Drivers","Multi-Data",
"Scenario","Risk","Decision","Export"
])

# =========================================================
# INSIGHTS
# =========================================================

with tabs[0]:
    if allow("eda"):
        col=st.selectbox("Metric",num_cols)
        st.plotly_chart(px.histogram(df,x=col),use_container_width=True)

# =========================================================
# KPI
# =========================================================

with tabs[1]:
    if allow("kpi"):
        m=st.selectbox("Metric",num_cols,key="kpi")
        c1,c2,c3=st.columns(3)
        c1.metric("Mean",round(df[m].mean(),2))
        c2.metric("Max",round(df[m].max(),2))
        c3.metric("Min",round(df[m].min(),2))

# =========================================================
# CORRELATION
# =========================================================

with tabs[2]:
    if allow("corr"):
        corr=df[num_cols].corr()
        st.plotly_chart(px.imshow(corr,text_auto=True),use_container_width=True)
    else: st.warning("Pro+")

# =========================================================
# ANOMALIES
# =========================================================

with tabs[3]:
    if allow("anomaly"):
        col=st.selectbox("Column",num_cols,key="a")
        X=df[[col]].dropna()
        model=IsolationForest(contamination=0.05)
        preds=model.fit_predict(X)
        X["flag"]=preds
        st.plotly_chart(px.scatter(X,y=col,color="flag"),use_container_width=True)
    else: st.warning("Pro+")

# =========================================================
# FORECAST
# =========================================================

with tabs[4]:
    if allow("forecast"):
        if date_cols:
            d=st.selectbox("Date",date_cols)
            v=st.selectbox("Value",num_cols)
            data=df[[d,v]].rename(columns={d:"ds",v:"y"}).dropna()

            m=Prophet()
            m.fit(data)
            fut=m.make_future_dataframe(periods=30)
            f=m.predict(fut)
            st.plotly_chart(px.line(f,x="ds",y="yhat"),use_container_width=True)
        else:
            st.error("No date column")

# =========================================================
# DRIVERS
# =========================================================

with tabs[5]:
    if allow("drivers"):
        target=st.selectbox("Target",num_cols)
        X=df[num_cols].drop(columns=[target]).dropna()
        y=df[target].loc[X.index]

        model=LinearRegression().fit(X,y)
        imp=pd.Series(model.coef_,index=X.columns).sort_values()

        st.plotly_chart(px.bar(imp,title="Driver Impact"),use_container_width=True)
    else: st.warning("Pro+")

# =========================================================
# MULTI DATA
# =========================================================

with tabs[6]:
    if allow("multidata"):
        if len(st.session_state.datasets)>1:
            metric=st.selectbox("Metric",num_cols)
            comp={}
            for n,d in st.session_state.datasets.items():
                if metric in d.columns:
                    comp[n]=d[metric].mean()
            st.plotly_chart(px.bar(x=list(comp.keys()),y=list(comp.values())))
        else:
            st.info("Upload >1 dataset")
    else: st.warning("Pro+")

# =========================================================
# SCENARIO
# =========================================================

with tabs[7]:
    if allow("scenario"):
        m=st.selectbox("Metric",num_cols,key="s")
        factor=st.slider("Change %",-100,100,10)/100
        sim=df[m]*(1+factor)
        fig=go.Figure()
        fig.add_scatter(y=df[m],name="Base")
        fig.add_scatter(y=sim,name="Scenario")
        st.plotly_chart(fig,use_container_width=True)
    else: st.warning("Enterprise only")

# =========================================================
# RISK ENGINE
# =========================================================

with tabs[8]:
    if allow("risk"):
        m=st.selectbox("Metric",num_cols,key="r")
        vol=df[m].std()
        score=min(100,vol*10)
        st.metric("Risk Score",round(score,2))
    else: st.warning("Enterprise only")

# =========================================================
# DECISION ENGINE
# =========================================================

with tabs[9]:
    if allow("decision"):
        m=st.selectbox("Metric",num_cols,key="d")
        trend=df[m].tail(10).mean()-df[m].head(10).mean()

        if trend>0:
            st.success("Recommendation: Increase investment — positive trend detected.")
        else:
            st.error("Recommendation: Investigate decline driver.")
    else: st.warning("Enterprise only")

# =========================================================
# PDF EXPORT
# =========================================================

with tabs[10]:
    if allow("pdf"):

        if st.button("Generate Executive PDF"):
            file=tempfile.NamedTemporaryFile(delete=False,suffix=".pdf")
            doc=SimpleDocTemplate(file.name,pagesize=letter)
            styles=getSampleStyleSheet()
            elements=[]

            elements.append(Paragraph("Executive Report",styles["Title"]))
            elements.append(Spacer(1,12))

            elements.append(Paragraph(f"Dataset: {dataset_name}",styles["Normal"]))
            elements.append(Paragraph(f"Rows: {df.shape[0]}",styles["Normal"]))
            elements.append(Paragraph(f"Columns: {df.shape[1]}",styles["Normal"]))

            doc.build(elements)

            with open(file.name,"rb") as f:
                st.download_button("Download PDF",f,"report.pdf")

    else: st.warning("Enterprise only")
