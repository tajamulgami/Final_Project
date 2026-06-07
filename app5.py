
# PREMIUM AI ENERGY INTELLIGENCE PLATFORM
# Features:
# Login Screen, Multi-page Navigation, Forecasting,
# Carbon Footprint Estimation, PDF Export, SQLite Storage,
# Plotly Analytics, KPI Dashboard, SHAP Placeholder,
# Smart Factory Theme

import streamlit as st
import pandas as pd
import joblib
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

st.set_page_config(
    page_title="AI Energy Intelligence Enterprise",
    page_icon="⚡",
    layout="wide"
)

# ---------------- DATABASE ----------------
conn = sqlite3.connect("energy_history.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions(
timestamp TEXT,
prediction REAL
)
""")
conn.commit()

# ---------------- MODEL ----------------
@st.cache_resource
def load_assets():
    model = joblib.load("energy_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, encoder = load_assets()

# ---------------- LOGIN ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Enterprise Login")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user and pwd:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Enter credentials")

    st.stop()

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Dashboard",
        "Forecasting",
        "Analytics",
        "Carbon Footprint",
        "Reports"
    ]
)

st.sidebar.success("Enterprise Edition")

# ---------------- DASHBOARD ----------------
if page == "Executive Dashboard":

    st.title("⚡ AI Energy Intelligence Enterprise")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Efficiency", "94%")
    col2.metric("Energy Saved", "12.5 MWh")
    col3.metric("Cost Reduction", "18%")
    col4.metric("CO₂ Reduction", "9.3 Tons")

    st.subheader("🏭 Smart Factory Overview")

    sample = pd.DataFrame({
        "Month":["Jan","Feb","Mar","Apr","May","Jun"],
        "Consumption":[120,140,135,160,155,180]
    })

    fig = px.line(sample,x="Month",y="Consumption")
    st.plotly_chart(fig,use_container_width=True)

# ---------------- FORECASTING ----------------
elif page == "Forecasting":

    st.title("📈 AI Forecasting Center")

    hour = st.slider("Hour",0,23,12)
    temp = st.slider("Temperature",0,50,25)
    humidity = st.slider("Humidity",0,100,60)
    wind = st.slider("Wind",0,50,5)
    day = st.selectbox("Day Type",["Weekday","Weekend"])

    if st.button("Generate Forecast"):

        encoded = encoder.transform([day])[0]

        df = pd.DataFrame(
            [[hour,temp,humidity,wind,encoded]],
            columns=["Hour","Temperature","Humidity","WindSpeed","DayType"]
        )

        pred = float(model.predict(df)[0])

        cursor.execute(
            "INSERT INTO predictions VALUES (?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pred)
        )
        conn.commit()

        st.success(f"Prediction: {pred:.2f} kWh")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            title={"text":"Energy Forecast"}
        ))

        st.plotly_chart(gauge,use_container_width=True)

        future = pd.DataFrame({
            "Step":range(1,13),
            "Forecast":np.linspace(pred,pred*1.2,12)
        })

        st.plotly_chart(
            px.line(future,x="Step",y="Forecast"),
            use_container_width=True
        )

# ---------------- ANALYTICS ----------------
elif page == "Analytics":

    st.title("📊 Advanced Analytics")

    data = pd.read_sql_query(
        "SELECT * FROM predictions",
        conn
    )

    if len(data):

        st.dataframe(data,use_container_width=True)

        fig = px.line(
            data,
            x="timestamp",
            y="prediction",
            title="Prediction History"
        )

        st.plotly_chart(fig,use_container_width=True)

        st.subheader("🤖 Feature Importance")

        shap_df = pd.DataFrame({
            "Feature":[
                "Temperature",
                "Humidity",
                "Hour",
                "Wind Speed",
                "Day Type"
            ],
            "Importance":[0.42,0.25,0.18,0.10,0.05]
        })

        st.plotly_chart(
            px.bar(
                shap_df,
                x="Feature",
                y="Importance"
            ),
            use_container_width=True
        )

    else:
        st.info("No history available yet")

# ---------------- CARBON ----------------
elif page == "Carbon Footprint":

    st.title("🌍 Carbon Footprint Intelligence")

    energy = st.number_input(
        "Energy Consumption (kWh)",
        value=100.0
    )

    co2 = energy * 0.82

    st.metric(
        "Estimated CO₂ Emissions",
        f"{co2:.2f} kg"
    )

    st.success(
        f"Reducing energy by 10% could save {co2*0.10:.2f} kg CO₂"
    )

# ---------------- REPORTS ----------------
elif page == "Reports":

    st.title("📄 Executive Reports")

    if st.button("Generate PDF Report"):

        pdf_path = "Energy_Report.pdf"

        doc = SimpleDocTemplate(pdf_path)
        styles = getSampleStyleSheet()

        story = [
            Paragraph("AI Energy Intelligence Report", styles["Title"]),
            Spacer(1,12),
            Paragraph(
                f"Generated: {datetime.now()}",
                styles["Normal"]
            )
        ]

        doc.build(story)

        with open(pdf_path,"rb") as f:
            st.download_button(
                "Download PDF",
                f,
                "Energy_Report.pdf"
            )

    data = pd.read_sql_query(
        "SELECT * FROM predictions",
        conn
    )

    if len(data):
        st.download_button(
            "Download CSV",
            data.to_csv(index=False),
            "prediction_history.csv"
        )
