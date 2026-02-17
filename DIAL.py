import streamlit as st

# =========================================================
# LOGIN GATE
# =========================================================

def check_login():

    if "logged" not in st.session_state:
        st.session_state.logged=False

    if st.session_state.logged:
        return True

    st.title("DIAL Secure Access")

    user=st.text_input("Username")
    pwd=st.text_input("Password",type="password")

    if st.button("Login"):

        if (
            user==st.secrets["auth"]["username"]
            and pwd==st.secrets["auth"]["password"]
        ):
            st.session_state.logged=True
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

check_login()

# =========================================================
# IMPORTS
# =========================================================

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
"copilot":["Enterprise"],
"alerts":["Enterprise"]
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
# NATURAL LANGUAGE ENGINE
# =========================================================

def answer_question(question, df, numeric_cols):

    q=question.lower()

    if "trend" in q or "growing" in q or "declining" in q:
        return [
            f"{c} increasing" if df[c].tail(10).mean()>df[c].head(10).mean()
            else f"{c} decreasing"
            for c in numeric_cols
        ]

    if "volatile" in q or "risk" in q:
        vols={c:df[c].std() for c in numeric_cols}
        worst=max(vols,key=vols.get)
        return [f"Most volatile metric: {worst}"]

    if "driver" in q or "impact" in q:
        target=numeric_cols[0]
        X=df[numeric_cols].drop(columns=[target]).dropna()
        y=df[target].loc[X.index]
        model=LinearRegression().fit(X,y)
        imp=pd.Series(model.coef_,index=X.columns)
        return [f"Strongest driver of {target}: {imp.abs().idxmax()}"]

    if "recommend" in q or "should" in q:
        alerts=[]
        for c in numeric_cols:
            if df[c].std()>df[c].mean()*0.3:
                alerts.append(f"Monitor {c}: high volatility")
        return alerts or ["System stable"]

    return ["Question not recognized"]

# =========================================================
# ALERT ENGINE
# =========================================================

def generate_alerts(df,num):

    alerts=[]

    for c in num:

        recent=df[c].tail(10).mean()
        old=df[c].head(10).mean()

        if recent<old*0.85:
            alerts.append(f"Alert: {c} dropped significantly")

        if df[c].std()>df[c].mean()*0.4:
            alerts.append(f"Risk: {c} extremely volatile")

        if df[c].max()>df[c].mean()*2:
            alerts.append(f"Spike detected in {c}")

    return alerts

# =========================================================
# HEADER
# =========================================================

st.title("DIAL — Decision Intelligence Analytics Lab")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Control Center")

tier=st.sidebar.selectbox(
"License Tier ?",
TIERS,
help="Simulate license feature access"
)
st.session_state.tier=tier

uploaded=st.sidebar.file_uploader(
"Upload Dataset ?",
accept_multiple_files=True,
help="Upload CSV / Excel / JSON"
)

if st.sidebar.button("Generate Demo Dataset ?"):
    st.session_state.datasets={"Demo Dataset":demo_data()}

if uploaded:
    for f in uploaded:
        st.session_state.datasets[f.name]=load_file(f)

# =========================================================
# ADVANCED CONFIG
# =========================================================

with st.sidebar.expander("Advanced Configuration",False):

    contamination=st.slider("Anomaly Sensitivity",0.01,0.2,0.05)
    sim_range=st.slider("Scenario Range %",10,300,100)
    risk_scale=st.slider("Risk Multiplier",1,50,10)

# =========================================================
# DATA CHECK
# =========================================================

if not st.session_state.datasets:
    st.info("Upload or generate dataset")
    st.stop()

name=st.selectbox("Dataset ?",list(st.session_state.datasets.keys()))
df=st.session_state.datasets[name]
num,date=detect_cols(df)

# =========================================================
# SUMMARY
# =========================================================

if allow("summary"):
    st.subheader("Executive Summary")
    st.success(
        f"Rows {df.shape[0]} | Columns {df.shape[1]} | Numeric {len(num)} | Missing {df.isna().sum().sum()}"
    )

# =========================================================
# ALERTS PANEL
# =========================================================

if allow("alerts"):
    alerts=generate_alerts(df,num)
    if alerts:
        st.error("Autonomous Alerts")
        for a in alerts:
            st.write("•",a)

# =========================================================
# TABS
# =========================================================

tabs=st.tabs([
"Insights","KPIs","Correlation","Anomalies",
"Forecast","Drivers","Scenario",
"Risk","Decision","AI Copilot","Export"
])

# INSIGHTS
with tabs[0]:
    if allow("eda"):
        m=st.selectbox("Metric ?",num)
        st.plotly_chart(px.histogram(df,x=m),use_container_width=True)

# KPI
with tabs[1]:
    if allow("kpi"):
        m=st.selectbox("Metric ?",num,key="k")
        c1,c2,c3=st.columns(3)
        c1.metric("Mean",round(df[m].mean(),2))
        c2.metric("Max",round(df[m].max(),2))
        c3.metric("Min",round(df[m].min(),2))

# CORR
with tabs[2]:
    if allow("corr"):
        st.plotly_chart(px.imshow(df[num].corr(),text_auto=True),use_container_width=True)

# ANOMALY
with tabs[3]:
    if allow("anomaly"):
        m=st.selectbox("Metric ?",num,key="a")
        X=df[[m]].dropna()
        iso=IsolationForest(contamination=contamination)
        X["flag"]=iso.fit_predict(X)
        st.plotly_chart(px.scatter(X,y=m,color="flag"),use_container_width=True)

# FORECAST
with tabs[4]:
    if allow("forecast") and date:
        d=st.selectbox("Date ?",date)
        v=st.selectbox("Value ?",num)
        data=df[[d,v]].rename(columns={d:"ds",v:"y"}).dropna()
        model=Prophet().fit(data)
        future=model.make_future_dataframe(periods=30)
        f=model.predict(future)
        st.plotly_chart(px.line(f,x="ds",y="yhat"),use_container_width=True)

# DRIVERS
with tabs[5]:
    if allow("drivers"):
        t=st.selectbox("Target ?",num)
        X=df[num].drop(columns=[t]).dropna()
        y=df[t].loc[X.index]
        reg=LinearRegression().fit(X,y)
        imp=pd.Series(reg.coef_,index=X.columns).sort_values()
        st.plotly_chart(px.bar(imp),use_container_width=True)

# SCENARIO
with tabs[6]:
    if allow("scenario"):
        m=st.selectbox("Metric ?",num,key="s")
        factor=st.slider("Impact %",-sim_range,sim_range,10)/100
        sim=df[m]*(1+factor)
        fig=go.Figure()
        fig.add_scatter(y=df[m],name="Actual")
        fig.add_scatter(y=sim,name="Scenario")
        st.plotly_chart(fig,use_container_width=True)

# RISK
with tabs[7]:
    if allow("risk"):
        m=st.selectbox("Metric ?",num,key="r")
        st.metric("Risk Score",round(min(100,df[m].std()*risk_scale),2))

# DECISION
with tabs[8]:
    if allow("decision"):
        m=st.selectbox("Metric ?",num,key="d")
        trend=df[m].tail(10).mean()-df[m].head(10).mean()
        st.success("Increase investment") if trend>0 else st.error("Investigate decline")

# COPILOT
with tabs[9]:
    if allow("copilot"):
        q=st.text_input("Ask question")
        if q:
            for a in answer_question(q,df,num):
                st.write("•",a)

# EXPORT
with tabs[10]:
    if allow("pdf"):
        if st.button("Generate PDF"):
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
                st.download_button("Download",f,"report.pdf")
