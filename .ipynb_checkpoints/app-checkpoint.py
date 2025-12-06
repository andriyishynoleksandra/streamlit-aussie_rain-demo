import streamlit as st
import pandas as pd
import joblib

st.set_page_config(layout="wide")

# -------------------- LOAD MODEL --------------------
model = joblib.load("model/aussie_rain.joblib")
pre = joblib.load("model/preprocessor.joblib")   # ⬅️ load preprocessing dict

num_cols = pre["num_cols"]
cat_cols = pre["cat_cols"]
dummy_columns = pre["dummy_columns"]

# -------------------- HEADER ------------------------
st.title("🌦️ Прогноз дощу в Австралії")

col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
with col_img2:
    st.image("images/rain_forecast.png", width=450)

# -------------------- LOCATION PICKER --------------------
st.subheader("📍 Виберіть місто")

cities = [
    "Sydney", "Melbourne", "Brisbane", "Adelaide", "Perth",
    "Hobart", "Canberra", "Darwin", "AliceSprings", "Cairns",
    "GoldCoast", "Wollongong", "Townsville", "Newcastle", "Ballarat"
]

Location = st.selectbox("Місто", cities, index=0)
st.markdown("---")

# -------------------- USER INPUT (5 FEATURES) ------------------------
st.subheader("🔧 Вхідні параметри погоди")

col1, col2 = st.columns(2)

with col1:
    Humidity3pm = st.number_input("Вологість о 15:00 (%)", value=50.0)
    Pressure3pm = st.number_input("Атмосферний тиск о 15:00 (hPa)", value=1010.0)
    Sunshine = st.number_input("Сонячне сяйво за добу (години)", value=8.0)

with col2:
    Humidity9am = st.number_input("Вологість о 09:00 (%)", value=65.0)
    Pressure9am = st.number_input("Атмосферний тиск о 09:00 (hPa)", value=1012.0)
    WindGustSpeed = st.number_input("Пориви вітру (км/год)", value=40.0)

# -------------------- DEFAULT VALUES FOR OTHER FEATURES --------------------
default_row = {
    "Date": "2020-01-01",
    "Location": Location,
    "WindGustDir": "N",
    "WindDir9am": "N",
    "WindDir3pm": "N",
    "Evaporation": 5.0,
    "WindGustSpeed": WindGustSpeed,
    "WindSpeed9am": 15.0,
    "WindSpeed3pm": 20.0,
    "Cloud9am": 4.0,
    "Cloud3pm": 4.0,
    "Temp9am": 18.0,
    "Temp3pm": 25.0,
    "MaxTemp": 26.0,
    "MinTemp": 15.0,
    "Rainfall": 1.0,
    "RainToday": "No",
    # User inputs:
    "Humidity3pm": Humidity3pm,
    "Humidity9am": Humidity9am,
    "Pressure3pm": Pressure3pm,
    "Pressure9am": Pressure9am,
    "Sunshine": Sunshine
}

input_data = pd.DataFrame([default_row])

# -------------------- PREDICTION BUTTON ------------------------
st.markdown("---")

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    clicked = st.button("🔍 ОТРИМАТИ ПРОГНОЗ", use_container_width=True)

# Custom CSS
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #2E86C1;
        color: white;
        padding: 18px 24px;
        font-size: 22px;
        border-radius: 10px;
        font-weight: 600;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background-color: #1B4F72;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- PREPROCESSING + PREDICT ------------------------
if clicked:

    # 1. Fill missing numeric/cat values
    input_data[num_cols] = input_data[num_cols].fillna(pre["num_medians"])
    input_data[cat_cols] = input_data[cat_cols].fillna(pre["cat_modes"])

    # 2. One-hot encode manually
    input_encoded = pd.get_dummies(input_data)

    # 3. Align columns to training model
    input_encoded = input_encoded.reindex(columns=dummy_columns, fill_value=0)

    # 4. Predict
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    # ------------------ DISPLAY RESULT -------------------
    st.subheader("📊 Результат прогнозування")

    if prediction == 1:
        st.success(f"🌧️ **Ймовірність дощу завтра: {probability:.2%}**")
    else:
        st.info(f"☀️ **Ймовірність дощу завтра низька: {probability:.2%}**")
