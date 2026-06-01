import streamlit as st
import pandas as pd
import joblib

# Load saved model and encoder
model = joblib.load("energy_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Page Configuration
st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Energy Consumption Prediction")
st.write("Predict energy consumption based on environmental conditions.")

# User Inputs
hour = st.slider("Hour of Day", 0, 23, 12)

temperature = st.number_input(
    "Temperature (°C)",
    min_value=0.0,
    max_value=100.0,
    value=25.0,
    step=5.0
)

humidity = st.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=55.0,
    step=5.0
)

windspeed = st.number_input(
    "Wind Speed",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
    step=5.0
)

day_type = st.selectbox(
    "Day Type",
    ["Weekday", "Weekend"]
)

# Encode Day Type
day_type_encoded = encoder.transform([day_type])[0]

# Show Input Summary
st.subheader("Input Summary")
st.write(f"Hour: {hour}")
st.write(f"Temperature: {temperature} °C")
st.write(f"Humidity: {humidity} %")
st.write(f"Wind Speed: {windspeed}")
st.write(f"Day Type: {day_type}")

# Predict Button
if st.button("Predict Energy Consumption"):

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

    st.success(
        f"⚡ Predicted Energy Consumption: {prediction[0]:.2f} kWh"
    )

# Footer
st.markdown("---")
st.write("Developed using Machine Learning and Streamlit")