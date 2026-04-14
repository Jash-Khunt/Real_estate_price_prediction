# Real Estate Price Prediction using Machine Learning

This project implements a comprehensive real estate price prediction system using various Machine Learning techniques. It includes a complete data science pipeline—from exploratory data analysis and feature engineering to model training, evaluation, and an interactive web-based dashboard.

## 📋 Project Overview
Traditional property valuation methods often fail to capture complex relationships between location, infrastructure, amenities, and market trends. This research explores how supervised machine learning models can provide accurate, scalable, and data-driven predictions for residential properties.

### Highlights:
- **Dataset**: House Price India (14,619 records, 23 original features).
- **Models**: Linear Regression, Decision Tree Regressor, and Random Forest Regressor.
- **Features**: Deep feature engineering including house age, renovation status, basement ratios, and total area.
- **Web Interface**: A premium, dark-themed Flask dashboard for visualizing results and making real-time predictions.

---

## 🚀 How to Run

### 1. Prerequisites
Ensure you have Python 3.9+ installed and the necessary libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn flask
```

### 2. Run the ML Pipeline (Standalone Analysis)
This script performs the EDA, trains the models, and saves high-quality plots to the `output_plots/` directory.
```bash
python real_estate_prediction.py
```

### 3. Run the Web Dashboard
This starts the Flask server and hosts the interactive dashboard.
```bash
python app.py
```
After running, open your browser and navigate to: **`http://127.0.0.1:5050`**

---

## 📊 Model Performance

| Model | RMSE | R² Score | MAE |
| :--- | :--- | :--- | :--- |
| Linear Regression | 147,235.26 | 0.7187 | 102,729.70 |
| Decision Tree | 128,051.59 | 0.7872 | 80,926.81 |
| **Random Forest (Best)** | **102,512.49** | **0.8636** | **63,290.07** |

---

## 📁 Project Structure

```text
.
├── House Price India.csv       # Primary dataset used for training
├── Bengaluru_House_Data.csv    # Alternative dataset provided
├── real_estate_prediction.py   # Standalone ML pipeline and EDA script
├── app.py                      # Flask backend for the web dashboard
├── templates/
│   └── index.html              # Frontend dashboard (HTML/CSS/JS)
├── output_plots/               # Generated visualization plots
├── requirements.txt            # Python dependencies for deployment
├── Procfile                    # Deployment configuration
└── README.md                   # Project documentation
```

---

## 🔍 Research Questions Answered
- **RQ1**: Which algorithm provides the highest accuracy?
  - *Answer*: **Random Forest** achieved the highest R² score of 0.8636.
- **RQ2**: How do features influence property prices?
  - *Answer*: The most significant influencers are **Grade of the house**, **Lattitude**, **Living Area**, and **Total Square Footage**.
- **RQ3**: Can ML outperform traditional valuation?
  - *Answer*: Yes, nonlinear models like Random Forest capture complex patterns that basic statistical methods miss, providing much higher reliability.
- **RQ4**: Impact of preprocessing?
  - *Answer*: Removing extreme outliers (1st-99th percentile) and engineering features like `house_age` and `total_area` significantly boosted model stability.

---

## 🚀 Deployment Guide (Render)

To deploy this project to the web using **Render**, follow these steps:

### 1. Push to GitHub
Ensure your latest code and configuration files are pushed to your repository:
```bash
git add .
git commit -m "Prepare for deployment"
git push origin main
```

### 2. Create a New Web Service on Render
1.  Log in to [Render.com](https://render.com).
2.  Click **New +** and select **Web Service**.
3.  Connect your GitHub repository: `Real_estate_price_prediction`.
4.  Configure the service:
    - **Name**: `real-estate-prediction-dashboard`
    - **Environment**: `Python 3`
    - **Build Command**: `pip install -r requirements.txt`
    - **Start Command**: `gunicorn app:app`
5.  Click **Create Web Service**.

---

## 👤 Submission Details
- **Faculty**: Dr. Nirali Nanavati
- **Course**: R&I Assignment 1