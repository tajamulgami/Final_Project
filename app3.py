import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("energy_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, encoder = load_assets()

# -----------------------------
# FUTURISTIC CSS
# -----------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
background: linear-gradient(135deg,#0f172a,#111827,#1e293b);
color:white;
}

.hero{
padding:25px;
border-radius:20px;
background:rgba(255,255,255,0.05);
backdrop-filter: blur(15px);
border:1px solid rgba(255,255,255,0.15);
text-align:center;
}

.big-title{
font-size:3rem;
font-weight:800;
background:linear-gradient(90deg,#00e5ff,#7c4dff,#00ff95);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

.metric-card{
padding:15px;
border-radius:15px;
background:rgba(255,255,255,0.06);
border:1px solid rgba(255,255,255,0.1);
text-align:center;
}

.prediction{
padding:30px;
border-radius:20px;
background:linear-gradient(135deg,#00c853,#00e676);
color:black;
font-size:32px;
font-weight:bold;
text-align:center;
}

section[data-testid="stSidebar"]{
background:#0b1220;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class="hero">
<div class="big-title">⚡ AI Energy Intelligence Dashboard</div>
<p>Next-Generation Energy Consumption Forecasting Platform</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Smart Controls")

hour = st.sidebar.slider("Hour", 0, 23, datetime.now().hour)
temperature = st.sidebar.slider("Temperature (°C)", 0, 50, 25)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 60)
windspeed = st.sidebar.slider("Wind Speed", 0, 50, 5)

day_type = st.sidebar.radio(
    "Day Type",
    ["Weekday", "Weekend"]
)

# -----------------------------
# LIVE METRICS
# -----------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f'<div class="metric-card"><h3>🕒 Hour</h3><h2>{hour}</h2></div>', unsafe_allow_html=True)

with c2:
    st.markdown(f'<div class="metric-card"><h3>🌡 Temp</h3><h2>{temperature}°C</h2></div>', unsafe_allow_html=True)

with c3:
    st.markdown(f'<div class="metric-card"><h3>💧 Humidity</h3><h2>{humidity}%</h2></div>', unsafe_allow_html=True)

with c4:
    st.markdown(f'<div class="metric-card"><h3>🌬 Wind</h3><h2>{windspeed}</h2></div>', unsafe_allow_html=True)

st.write("")

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🚀 Generate AI Prediction", use_container_width=True):

    day_encoded = encoder.transform([day_type])[0]

    df = pd.DataFrame(
        [[hour, temperature, humidity, windspeed, day_encoded]],
        columns=["Hour","Temperature","Humidity","WindSpeed","DayType"]
    )

    prediction = model.predict(df)[0]

    st.markdown(
        f"""
        <div class="prediction">
        ⚡ Predicted Energy Consumption<br>
        {prediction:.2f} kWh
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(min(prediction/100,1.0))

    if prediction < 30:
        st.success("🟢 Low Energy Usage Expected")
    elif prediction < 70:
        st.warning("🟡 Moderate Energy Usage Expected")
    else:
        st.error("🔴 High Energy Usage Expected")

# -----------------------------
# INSIGHTS
# -----------------------------
st.markdown("### 🤖 AI Insights")

st.info("""
• Higher temperatures generally increase energy demand.

• Weekend and weekday patterns can differ significantly.

• Humidity and wind conditions influence consumption behavior.

• Use this dashboard for proactive energy planning.
""")

st.markdown("---")
st.caption("Powered by Machine Learning • Streamlit • Predictive Analytics")
