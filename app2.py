import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# Load Model & Encoder
# ------------------------------
model = joblib.load("energy_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide"
)

# ------------------------------
# Custom CSS
# ------------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    color: #1f4e79;
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    background-color: #d4edda;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    color: #155724;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Header
# ------------------------------
st.markdown(
    '<p class="title">⚡ Energy Consumption Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Predict future energy consumption using environmental conditions and time-based factors</p>',
    unsafe_allow_html=True
)

# ------------------------------
# Input Section
# ------------------------------
st.subheader("📊 Enter Input Parameters")

col1, col2 = st.columns(2)

with col1:
    hour = st.slider(
        "🕒 Hour of Day",
        min_value=0,
        max_value=23,
        value=12
    )

    temperature = st.number_input(
        "🌡 Temperature (°C)",
        min_value=0.0,
        max_value=100.0,
        value=25.0
    )

with col2:
    humidity = st.number_input(
        "💧 Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=55.0
    )

    windspeed = st.number_input(
        "🌬 Wind Speed",
        min_value=0.0,
        max_value=100.0,
        value=5.0
    )

day_type = st.selectbox(
    "📅 Day Type",
    ["Weekday", "Weekend"]
)

# ------------------------------
# Input Summary
# ------------------------------
with st.expander("📋 View Input Summary"):
    st.write(f"**Hour:** {hour}")
    st.write(f"**Temperature:** {temperature} °C")
    st.write(f"**Humidity:** {humidity}%")
    st.write(f"**Wind Speed:** {windspeed}")
    st.write(f"**Day Type:** {day_type}")

# ------------------------------
# Prediction
# ------------------------------
if st.button("🔮 Predict Energy Consumption", use_container_width=True):

    day_type_encoded = encoder.transform([day_type])[0]

    input_data = pd.DataFrame(
        [[hour, temperature, humidity, windspeed, day_type_encoded]],
        columns=[
            "Hour",
            "Temperature",
            "Humidity",
            "WindSpeed",
            "DayType"
        ]
    )

    prediction = model.predict(input_data)

    st.markdown(
        f"""
        <div class="result-box">
            ⚡ Predicted Energy Consumption<br>
            {prediction[0]:.2f} kWh
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")

st.markdown(
    """
    <div class="footer">
        Developed using <b>Machine Learning</b>, <b>Python</b>, and <b>Streamlit</b><br>
        Energy Consumption Prediction Capstone Project
    </div>
    """,
    unsafe_allow_html=True
)