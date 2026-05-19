import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="centered"
)

# Load dataset
df = pd.read_csv("data/crop_data.csv")

# Load saved model
model = joblib.load("models/crop_model.pkl")

# Title
st.title("🌱 Weather & Crop Recommendation System")

st.markdown("""
This AI system recommends the best crop based on:
- Soil nutrients
- Weather conditions
- Rainfall
""")

st.divider()

# Sidebar
st.sidebar.header("Project Information")

st.sidebar.write("""
Technology Used:
- Python
- Machine Learning
- Random Forest
- Streamlit
""")

# Inputs
st.subheader("Enter Agricultural Details")

N = st.number_input("Nitrogen (N)", min_value=0)
P = st.number_input("Phosphorus (P)", min_value=0)
K = st.number_input("Potassium (K)", min_value=0)

temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH Value")
rainfall = st.number_input("Rainfall (mm)")

# Prediction
if st.button("Predict Best Crop"):

    data = [[N, P, K, temperature, humidity, ph, rainfall]]

    prediction = model.predict(data)

    st.success(f"✅ Recommended Crop: {prediction[0]}")

    st.balloons()

# Dataset preview
st.divider()

if st.checkbox("Show Dataset Sample"):
    st.write(df.head())

    # -----------------------------
# Visualizations
# -----------------------------

st.divider()

st.subheader("📊 Data Visualizations")

# Crop distribution chart
st.write("### Crop Distribution")

crop_counts = df['label'].value_counts()

fig1, ax1 = plt.subplots(figsize=(10,5))

crop_counts.plot(kind='bar', ax=ax1)

plt.xlabel("Crop")
plt.ylabel("Count")
plt.title("Crop Distribution")

st.pyplot(fig1)

# Temperature distribution
st.write("### Temperature Distribution")

fig2, ax2 = plt.subplots(figsize=(8,5))

ax2.hist(df['temperature'], bins=20)

plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.title("Temperature Distribution")

st.pyplot(fig2)