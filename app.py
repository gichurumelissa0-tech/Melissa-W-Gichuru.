import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import seaborn as sns
from rapidfuzz import process, fuzz
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, time
import matplotlib.pyplot as plt
import sqlite3

# Paths & model loading
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "data", "Tv DataSet.csv")
model_dir = os.path.join(base_dir, "models")

os.makedirs(model_dir, exist_ok=True)

# Load trained models 
try:
    view_rf = joblib.load(os.path.join(model_dir, "rf_viewers.joblib")) 
    rev_rf = joblib.load(os.path.join(model_dir, "rf_revenue.joblib")) 
    encoders = joblib.load(os.path.join(model_dir, "encoders.joblib"))
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

le_show = encoders['le_show']
le_type = encoders['le_type']
day_map = encoders['day_map']
pop_map = encoders['pop_map']

# Load dataset for analytics
data = pd.read_csv( data_path )

def init_db():
    conn = sqlite3.connect("forecast_history.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            show_name TEXT,
            show_type TEXT,
            day TEXT,
            viewers INTEGER,
            revenue REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_prediction_to_db(show, show_type, day, viewers, revenue):
    conn = sqlite3.connect("forecast_history.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO history (show_name, show_type, day, viewers, revenue, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        show,
        show_type,
        day,
        int(viewers),
        float(revenue),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

def load_prediction_history():
    conn = sqlite3.connect("forecast_history.db")
    df = pd.read_sql("SELECT * FROM history ORDER BY timestamp DESC", conn)
    conn.close()
    return df

# Initialize DB at app start
init_db()

# Streamlit Page Config
st.set_page_config(
    page_title="TV SHOW PREDICTION APP",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>
.hero-banner { background-color: #1f77b4; color: white; padding: 2rem; border-radius: 10px; text-align: center; margin-bottom: 20px; }
.card { background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 1rem; text-align: center; }
.sidebar .sidebar-content { background-color: #f0f2f6; }
.stPyplot { border-radius: 10px; box-shadow: 2px 2px 12px rgba(0,0,0,0.1); padding: 10px; background-color: #f8f9fa; }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict", "Prediction Analytics", "Analytics", "Upload", "Chat"])


# ----------------------
# FEATURE ENGINEERING FUNCTION
# ----------------------
def create_features(show_name, type_show, popularity, start_time, end_time, day):
    start_hour = start_time.hour + start_time.minute/60
    end_hour = end_time.hour + end_time.minute/60

    duration = (datetime.combine(datetime.today(), end_time) -
                datetime.combine(datetime.today(), start_time)).seconds / 60

    if duration < 0:
        duration += 24 * 60

    # Cyclical encoding
    start_sin = np.sin(2 * np.pi * start_hour / 24)
    start_cos = np.cos(2 * np.pi * start_hour / 24)

    day_num = day_map[day]
    day_sin = np.sin(2 * np.pi * day_num / 7)
    day_cos = np.cos(2 * np.pi * day_num / 7)

    # Encodings
    show_label = le_show.transform([show_name])[0]
    type_label = le_type.transform([type_show])[0]
    pop_num = pop_map[popularity]

    # New Features (IMPROVEMENT)
    is_prime_time = 1 if 19 <= start_hour <= 22 else 0
    is_weekend = 1 if day in ["Saturday", "Sunday"] else 0
    interaction = duration * pop_num

    return np.array([[
        show_label, type_label, duration, pop_num,
        start_sin, start_cos, day_num, day_sin, day_cos,
        is_prime_time, is_weekend, interaction
    ]])


# ----------------------
# HOME PAGE
# ----------------------
if page == "Home":
    st.markdown('<div class="hero-banner"><h1>TV Forecast Dashboard</h1><p>Predict Viewers and Ad Revenue</p></div>', unsafe_allow_html=True)
    video_path = #insert video path.
    if os.path.exists(video_path):
        st.video(video_path)
    else:
        st.info("Hero video not found. Place your video at the correct path to display it.")

# ----------------------
# PREDICT PAGE
# ----------------------
elif page == "Predict":
    st.title("Predict TV Show Viewers & Ad Revenue")

    # User Inputs
    show_name = st.selectbox("Show Name", le_show.classes_)
    type_show = st.selectbox("Type of Show", le_type.classes_)
    popularity = st.selectbox("Popularity", list(pop_map.keys()))
    start_time = st.time_input("Start Time", time(20,0))
    end_time = st.time_input("End Time", time(21,0))
    day = st.selectbox("Day of Week", list(day_map.keys()))

    # Feature Engineering
    start_hour = start_time.hour + start_time.minute/60
    end_hour = end_time.hour + end_time.minute/60
    duration = (datetime.combine(datetime.today(), end_time) - datetime.combine(datetime.today(), start_time)).seconds / 60
    if duration < 0:
        duration += 24*60  # handle past-midnight

    start_sin = np.sin(2*np.pi*start_hour/24)
    start_cos = np.cos(2*np.pi*start_hour/24)
    day_num = day_map[day]
    day_sin = np.sin(2*np.pi*day_num/7)
    day_cos = np.cos(2*np.pi*day_num/7)

    show_label = le_show.transform([show_name])[0]
    type_label = le_type.transform([type_show])[0]
    pop_num = pop_map[popularity]

    X_input = np.array([[show_label, type_label, duration, pop_num, start_sin, start_cos, day_num, day_sin, day_cos]])

    # Predict button
    if st.button("Predict"):
        try:
            # Predictions
            viewers_pred = view_rf.predict(X_input)[0]
            X_rev = np.array([[show_label, type_label, duration, pop_num, start_sin, start_cos, day_num, day_sin, day_cos, viewers_pred]])
            revenue_pred = rev_rf.predict(X_rev)[0]

            # Save to SQLite
            save_prediction_to_db(show_name, type_show, day, viewers_pred, revenue_pred)

            # Display predictions
            col1, col2 = st.columns(2)
            with col1: 
                st.markdown(f'<div class="card"><h3>Predicted Viewers</h3><p style="font-size:24px">{int(viewers_pred):,}</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="card"><h3>Predicted Ad Revenue</h3><p style="font-size:24px">KSH{revenue_pred:,.0f}</p></div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ----------------------
# PREDICTION ANALYTICS
# ----------------------
elif page == "Prediction Analytics":
    st.title("Prediction Analytics")
    pred_df = load_prediction_history()

    if pred_df.empty:
        st.info("No predictions made yet.")
        st.stop()

    st.subheader("Prediction History")
    st.dataframe(pred_df)

    # Aggregate by show
    pred_show_agg = pred_df.groupby("show_name").agg({
        "viewers": "mean",
        "revenue": "mean"
    }).reset_index()

    # Average Revenue per Show
    st.subheader("Average Predicted Ad Revenue per Show")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(
        data=pred_show_agg,
        x="show_name",
        y="revenue",
        palette="viridis",
        order=pred_show_agg.sort_values("revenue", ascending=False)["show_name"],
        ax=ax
    )
    ax.set_ylabel("Average Predicted Revenue")
    ax.set_xlabel("Show Name")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)

    # Average Viewers per Show
    st.subheader("Average Predicted Viewers per Show")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(
        data=pred_show_agg,
        x="show_name",
        y="viewers",
        palette="coolwarm",
        order=pred_show_agg.sort_values("viewers", ascending=False)["show_name"],
        ax=ax
    )
    ax.set_ylabel("Average Predicted Viewers")
    ax.set_xlabel("Show Name")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)

    # Scatter plot: Viewers vs Revenue
    st.subheader("Predicted Viewers vs Predicted Revenue")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(pred_df["viewers"], pred_df["revenue"], color="#1f77b4", alpha=0.7)
    ax.set_xlabel("Predicted Viewers")
    ax.set_ylabel("Predicted Revenue")
    ax.set_title("Viewers vs Revenue (Predictions)")
    st.pyplot(fig)

# ----------------------
# ANALYTICS PAGE
# ----------------------
elif page == "Analytics":
    st.title("Data Analytics & Charts")

    # Revenue by Show Type
    st.subheader("Revenue by Show Type")
    type_revenue = data.groupby("Type_of_show")["Ad_revenue"].sum()
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x=type_revenue.index, y=type_revenue.values, palette="magma", ax=ax)
    ax.set_ylabel("Total Ad Revenue")
    st.pyplot(fig)

    # Viewers vs Revenue
    st.subheader("Viewers vs Ad Revenue")
    fig, ax = plt.subplots()
    ax.scatter(data['Viewers'], data['Ad_revenue'], alpha=0.6, color="#1f77b4")
    ax.set_xlabel("Viewers")
    ax.set_ylabel("Ad Revenue")
    ax.set_title("Viewers vs Ad Revenue")
    st.pyplot(fig)

# ----------------------
# UPLOAD PAGE
# ----------------------
elif page == "Upload":
    st.title("Upload Your CSV Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        uploaded_data = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(uploaded_data.head())
        st.success("File uploaded successfully!")

# ----------------------
# CHAT PAGE
# ----------------------
elif page == "Chat":
    st.header("Analytics Chatbot 🤖")

    user_input = st.text_input("Ask a question:")

    if st.button("Ask") and user_input:

        msg = user_input.lower()

        if "top shows" in msg:
            top = data.groupby("Show_name")["Viewers"].mean().sort_values(ascending=False).head(5)
            st.write(top)

        elif "best revenue" in msg:
            top = data.groupby("Show_name")["Ad_revenue"].mean().sort_values(ascending=False).head(5)
            st.write(top)

        elif "average revenue" in msg:
            st.write(f"Average Revenue: KSH {data['Ad_revenue'].mean():,.0f}")

        elif "average viewers" in msg:
            st.write(f"Average Viewers: {data['Viewers'].mean():,.0f}")

        else:
            st.warning("Try: top shows, best revenue, average viewers, etc.")