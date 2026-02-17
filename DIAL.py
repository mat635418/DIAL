import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from prophet import Prophet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(page_title="DIAL Platform", layout="wide")

# =========================================================
# TIERS
# =========================================================

TIERS=["Starter","Pro","Enterprise"]

ACCESS={
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
"decision":["Enterprise"],
"copilot":["Enterprise"]
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
    if file.name.endswith("csv"): return pd.read_csv(file)
    if file.name.endswith("xlsx"): return pd.read_excel(file)
    return pd.read_json(file)

def detect_cols(df):
    nums=df.select_dtypes(include=np.number).columns.tolist()
    dates=[]
    for c in df.columns:
        if "date" in c.lower():
            try:
                df[c]=pd.to_datetime(df[c])
                dates.append(c)
            except: pass
    return nums,dates

def demo_data():
    np.random.seed(1)
    d=pd.date_range("2024-01-01",periods=200)
    return pd.DataFrame({
        "date":d,
        "revenue":np.random.normal(500,40,200).cumsum(),
        "cost":np.random.normal(300,25,200).cumsum(),
        "inventory":np.random.randint(50,200,200),
        "demand":np.random.normal(100,15,200)
    })

# =========================================================
# HEADER
# =========================================================

st.title("DIAL — Decision Intelligence Analytics Lab")

# =========================================================
# SIDEBAR CONTROL CENTER
# =========================================================

st.sidebar.header("Control Center")

tier=st.sidebar.selectbox(
"License Tier ?",
TIERS,
help="Select license to simulate feature access levels"
)
st.session_state.tier=tier

uploaded=st.sidebar.file_uploader(
"Upload Dataset ?",
accept_multiple_files=True,
help="Upload CSV, Excel, or JSON files. Numeric columns enable analytics."
)

if st.sidebar.button("Generate Demo Dataset ?",
help="Loads sample dataset for live demonstration"):
    st.session_state.datasets={"Demo Dataset":demo_data()}

# load uploads
if uploaded:
    for f in uploaded:
        st.session_state.datasets[f.name]=load_file(f)

# =========================================================
# ADVANCED CONFIG PANEL
# =========================================================

with st.sidebar.expander("Advanced Configuration", expanded=False):

    st.slider(
        "Forecast Horizon",
        7,180,30,
        help="Number of future periods forecast models should predict"
    )

    contamination=st.slider(
        "Anomaly Sensitivity",
        0.01,0.2,0.05,
        help="Higher value = more anomalies detected"
    )

    sim_range=st.slider(
        "Scenario Max Impact %",
        10,300,100,
        help="Maximum simulation adjustment range"
    )

    risk_scale=st.slider(
        "Risk Multiplier",
        1,50,10,
        help="Adjusts how aggressively risk score scales"
    )

# =========================================================
# DATA CHECK
# =========================================================

if not st.session_state.datasets:
    st.info("Upload or generate dataset to begin")
    st.stop()

name=st.selectbox(
"Select Dataset ?",
list(st.session_state.datasets.keys()),
help="Choose dataset to analyze"
)

df=st.session_state.datasets[name]
num,date=detect_cols(df)

# =========================================================
# SUMMARY
# =========================================================

if allow("summary"):
    st.subheader("Executive Summary ?",
    help="High level dataset diagnostic overview")

    st.success(
        f"Rows {df.shape[0]} | Columns {df.shape[1]} | Numeric {len(num)} | Missing {df.isna().sum().sum()}"
    )

# =========================================================
# TABS
# =========================================================

tabs=st.tabs([
"Insights","KPIs","Correlation","Anomalies",
"Forecast","Drivers","MultiData",
"Scenario","Risk","Decision","AI Copilot","Export"
])

# =========================================================
# INSIGHTS
# =========================================================

with tabs[0]:
    if allow("eda"):
        m=st.selectbox("Metric ?",num,help="Select variable to analyze distribution")
        st.plotly_chart(px.histogram(df,x=m),use_container_width=True)

# =========================================================
# KPI
# =========================================================

with tabs[1]:
    if allow("kpi"):
        m=st.selectbox("Metric ?",num,key="k",help="Choose metric for KPI stats")
        c1,c2,c3=st.columns(3)
        c1.metric("Mean",round(df[m].mean(),2))
        c2.metric("Max",round(df[m].max(),2))
        c3.metric("Min",round(df[m].min(),2))

# =========================================================
# CORRELATION
# =========================================================

with tabs[2]:
    if allow("corr"):
        st.plotly_chart(px.imshow(df[num].corr(),text_auto=True),use_container_width=True)
    else: st.warning("Pro+")

# =========================================================
# ANOMALY
# =========================================================

with tabs[3]:
    if allow("anomaly"):
        m=st.selectbox("Metric ?",num,key="a",help="Detect outliers")
        X=df[[m]].dropna()
        iso=IsolationForest(contamination=contamination)
        X["flag"]=iso.fit_predict(X)
        st.plotly_chart(px.scatter(X,y=m,color="flag"),use_container_width=True)
    else: st.warning("Pro+")

# =========================================================
# FORECAST
# =========================================================

with tabs[4]:
    if allow("forecast"):
        if date:
            d=st.selectbox("Date ?",date)
            v=st.selectbox("Value ?",num)
            data=df[[d,v]].rename(columns={d:"ds",v:"y"}).dropna()

            model=Prophet()
            model.fit(data)

            future=model.make_future_dataframe(periods=30)
            f=model.predict(future)

            st.plotly_chart(px.line(f,x="ds",y="yhat"),use_container_width=True)
        else:
            st.error("No date column detected")

# =========================================================
# DRIVERS
# =========================================================

with tabs[5]:
    if allow("drivers"):
        t=st.selectbox("Target ?",num)
        X=df[num].drop(columns=[t]).dropna()
        y=df[t].loc[X.index]

        reg=LinearRegression().fit(X,y)
        imp=pd.Series(reg.coef_,index=X.columns).sort_values()

        st.plotly_chart(px.bar(imp,title="Driver Importance"),use_container_width=True)
    else: st.warning("Pro+")

# =========================================================
# MULTIDATA
# =========================================================

with tabs[6]:
    if allow("multidata"):
        if len(st.session_state.datasets)>1:
            m=st.selectbox("Metric ?",num,key="md")
            comp={n:d[m].mean() for n,d in st.session_state.datasets.items() if m in d}
            st.plotly_chart(px.bar(x=list(comp.keys()),y=list(comp.values())))
        else: st.info("Upload multiple datasets")
    else: st.warning("Pro+")

# =========================================================
# SCENARIO
# =========================================================

with tabs[7]:
    if allow("scenario"):
        m=st.selectbox("Metric ?",num,key="s")
        factor=st.slider("Impact %",-sim_range,sim_range,10)/100
        sim=df[m]*(1+factor)
        fig=go.Figure()
        fig.add_scatter(y=df[m],name="Actual")
        fig.add_scatter(y=sim,name="Scenario")
        st.plotly_chart(fig,use_container_width=True)
    else: st.warning("Enterprise only")

# =========================================================
# RISK
# =========================================================

with tabs[8]:
    if allow("risk"):
        m=st.selectbox("Metric ?",num,key="r")
        score=min(100,df[m].std()*risk_scale)
        st.metric("Risk Score",round(score,2))
    else: st.warning("Enterprise only")

# =========================================================
# DECISION ENGINE
# =========================================================

with tabs[9]:
    if allow("decision"):
        m=st.selectbox("Metric ?",num,key="d")
        trend=df[m].tail(10).mean()-df[m].head(10).mean()

        if trend>0:
            st.success("Increase investment — upward trend detected")
        else:
            st.error("Investigate decline drivers")
    else: st.warning("Enterprise only")

# =========================================================
# AI COPILOT
# =========================================================

with tabs[10]:
    if allow("copilot"):
        st.subheader("AI Copilot Insights ?",
        help="Automatically interprets data and provides executive guidance")

        insights=[]

        for c in num:
            trend=df[c].tail(10).mean()-df[c].head(10).mean()
            if trend>0:
                insights.append(f"{c} increasing trend")
            else:
                insights.append(f"{c} declining trend")

            if df[c].std()>df[c].mean()*0.3:
                insights.append(f"{c} highly volatile")

        for i in insights:
            st.write("•",i)

    else: st.warning("Enterprise only")

# =========================================================
# PDF EXPORT
# =========================================================

with tabs[11]:
    if allow("pdf"):
        if st.button("Generate Executive Report ?",
        help="Download board-ready PDF summary"):

            file=tempfile.NamedTemporaryFile(delete=False,suffix=".pdf")
            doc=SimpleDocTemplate(file.name,pagesize=letter)
            styles=getSampleStyleSheet()

            elements=[
                Paragraph("Executive Report",styles["Title"]),
                Spacer(1,12),
                Paragraph(f"Dataset: {name}",styles["Normal"]),
                Paragraph(f"Rows: {df.shape[0]}",styles["Normal"]),
                Paragraph(f"Columns: {df.shape[1]}",styles["Normal"])
            ]

            doc.build(elements)

            with open(file.name,"rb") as f:
                st.download_button("Download PDF",f,"report.pdf")

    else: st.warning("Enterprise only")
