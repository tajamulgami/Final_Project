
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

st.set_page_config(
    page_title="AI Energy Intelligence Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_assets():
    model = joblib.load("energy_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, encoder = load_assets()

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
background:linear-gradient(135deg,#0f172a,#111827,#1e293b);
}
.block-container{padding-top:1rem;}
</style>
""", unsafe_allow_html=True)

st.title("⚡ AI Energy Intelligence Pro Dashboard")
st.caption("Enterprise-Grade Energy Forecasting Platform")

with st.sidebar:
    st.header("🎛 Control Center")
    hour = st.slider("Hour",0,23,12)
    temperature = st.slider("Temperature",0,50,25)
    humidity = st.slider("Humidity",0,100,60)
    windspeed = st.slider("Wind Speed",0,50,5)
    day_type = st.selectbox("Day Type",["Weekday","Weekend"])
    dark_mode = st.toggle("Dark Mode", value=True)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Hour", hour)
c2.metric("Temperature", f"{temperature}°C")
c3.metric("Humidity", f"{humidity}%")
c4.metric("Wind", windspeed)

if st.button("🚀 Generate Prediction", use_container_width=True):
    day_encoded = encoder.transform([day_type])[0]

    df = pd.DataFrame([[hour,temperature,humidity,windspeed,day_encoded]],
                      columns=["Hour","Temperature","Humidity","WindSpeed","DayType"])

    prediction = float(model.predict(df)[0])

    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Prediction": prediction
    })

    st.success(f"Predicted Energy Consumption: {prediction:.2f} kWh")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={'text':"Energy Consumption"},
        gauge={'axis':{'range':[0,max(100,prediction+20)]}}
    ))
    st.plotly_chart(gauge, use_container_width=True)

    if prediction < 30:
        st.success("🟢 Low Consumption")
    elif prediction < 70:
        st.warning("🟡 Moderate Consumption")
    else:
        st.error("🔴 High Consumption")

tab1, tab2, tab3 = st.tabs(["📈 Trends","📋 History","🤖 AI Insights"])

with tab1:
    if st.session_state.history:
        hist = pd.DataFrame(st.session_state.history)
        fig = px.line(hist, x="Time", y="Prediction",
                      title="Prediction Trend")
        st.plotly_chart(fig, use_container_width=True)

        forecast = pd.DataFrame({
            "Future Step": list(range(1,11)),
            "Forecast": np.linspace(
                hist["Prediction"].iloc[-1],
                hist["Prediction"].iloc[-1]*1.15,
                10
            )
        })

        fig2 = px.line(forecast, x="Future Step", y="Forecast",
                       title="Forecast Simulation")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history),
                     use_container_width=True)

        csv = pd.DataFrame(st.session_state.history).to_csv(index=False)
        st.download_button(
            "📥 Download Report CSV",
            csv,
            "energy_prediction_report.csv",
            "text/csv"
        )

with tab3:
    st.info("""
    • High temperatures usually increase demand.

    • Humidity affects cooling efficiency.

    • Weekend patterns often differ from weekdays.

    • Forecasting helps optimize operational costs.

    • Use historical predictions for capacity planning.
    """)

st.markdown("---")
st.caption("AI Energy Intelligence Pro • Streamlit • Plotly • Machine Learning")
