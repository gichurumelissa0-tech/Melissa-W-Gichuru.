# TV Analytics & Revenue Forecasting System

A machine learning-powered Streamlit application that predicts **TV show viewership** and **advertising revenue** using historical broadcasting data.

This project demonstrates how **data analytics and machine learning** can be used to optimize TV programming, improve audience engagement, and maximize advertising revenue.

---

## Features

* Predict TV show **viewers and ad revenue**
*  Machine Learning models (Random Forest & Linear Regression)
*  Interactive dashboard built with Streamlit
*  Data visualizations for insights and trends
*  Stores prediction history using SQLite
*  Simple analytics chatbot
*  Upload your own dataset for analysis

---

## Machine Learning Approach

### Feature Engineering:

* Time-based cyclical encoding (sin/cos)
* Show duration calculation
* Show popularity scoring
* Day-of-week encoding

### Models Used:

* Random Forest Regressor (primary model)
* Linear Regression (baseline model)

### Predictions:

*  Viewers
*  Ad Revenue (based on predicted viewers)

---

## Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Machine Learning:** Scikit-learn
* **Database:** SQLite
* **Visualization:** Matplotlib, Seaborn
* **Data Processing:** Pandas, NumPy

---

## Project Structure

tv-forecast-dashboard/
│
├── app.py
├── model_training.py
├── data/
│   └── Tv DataSet.csv
├── models/
│   ├── rf_viewers.joblib
│   ├── rf_revenue.joblib
│   └── encoders.joblib
├── forecast_history.db
├── requirements.txt
└── README.md

---

## Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

### Required Libraries:

* streamlit
* pandas
* numpy
* scikit-learn
* joblib
* matplotlib
* seaborn
* rapidfuzz

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/tv-forecast-dashboard.git
cd tv-forecast-dashboard
```

### 2. Train the models (first time only)

```bash
python model_training.py
```

### 3. Run the app

```bash
streamlit run app.py
```

### 4. Open in browser

```
http://localhost:8501
```

---

## What This Project Does

This application helps answer key questions such as:

* Which TV shows attract the most viewers?
* What time slots generate the highest revenue?
* How does show popularity affect performance?
* What is the relationship between viewers and ad revenue?

---

## Insights Generated

* High-popularity shows during prime time yield higher revenue
* Viewer engagement varies by day and time
* Strong correlation between viewership and advertising revenue

---

## Future Improvements

* Deploy application online (Streamlit Cloud)
* Add real-time data integration
* Improve chatbot with NLP capabilities
* Use advanced models (e.g., XGBoost)
* Add recommendation system for optimal scheduling

---

## License

This project is for academic and educational purposes.
