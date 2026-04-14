"""
Flask backend for Real Estate Price Prediction Dashboard
"""
import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

app = Flask(__name__, static_folder="static", template_folder="templates")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "House Price India.csv")
PLOT_DIR   = os.path.join(BASE_DIR, "output_plots")

# ── Global model objects (trained once on startup) ──
trained_models = {}
scaler         = None
feature_cols   = []
model_metrics  = {}
feature_stats  = {}
dataset_info   = {}


def train_all_models():
    """Train all ML models and store them globally."""
    global trained_models, scaler, feature_cols, model_metrics, feature_stats, dataset_info

    df = pd.read_csv(DATA_PATH)
    original_rows = df.shape[0]

    # Drop identifier columns
    df.drop(columns=["id", "Date"], inplace=True, errors="ignore")

    # Feature engineering
    df["house_age"]      = 2026 - df["Built Year"]
    df["is_renovated"]   = (df["Renovation Year"] > 0).astype(int)
    df["basement_ratio"] = df["Area of the basement"] / (df["Area of the house(excluding basement)"] + 1)
    df["total_area"]     = df["Area of the house(excluding basement)"] + df["Area of the basement"]
    df["bed_bath_ratio"] = df["number of bedrooms"] / (df["number of bathrooms"] + 1)

    # Outlier removal
    Q1   = df["Price"].quantile(0.01)
    Q3   = df["Price"].quantile(0.99)
    df   = df[(df["Price"] >= Q1) & (df["Price"] <= Q3)].reset_index(drop=True)

    # Store dataset info
    dataset_info["total_rows"]    = original_rows
    dataset_info["clean_rows"]    = df.shape[0]
    dataset_info["num_features"]  = df.shape[1] - 1
    dataset_info["price_min"]     = float(df["Price"].min())
    dataset_info["price_max"]     = float(df["Price"].max())
    dataset_info["price_mean"]    = float(df["Price"].mean())
    dataset_info["price_median"]  = float(df["Price"].median())

    # Features / target
    drop_for_model = ["Price"]
    if "price_per_sqft" in df.columns:
        drop_for_model.append("price_per_sqft")
    feature_cols_local = [c for c in df.columns if c not in drop_for_model]
    feature_cols = feature_cols_local

    X = df[feature_cols].copy()
    y = df["Price"].copy()

    # Store feature stats for the prediction form
    for col in feature_cols:
        feature_stats[col] = {
            "min":    float(X[col].min()),
            "max":    float(X[col].max()),
            "mean":   float(X[col].mean()),
            "median": float(X[col].median()),
        }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_obj = StandardScaler()
    X_train_s  = scaler_obj.fit_transform(X_train)
    X_test_s   = scaler_obj.transform(X_test)
    scaler     = scaler_obj

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree":     DecisionTreeRegressor(max_depth=12, min_samples_split=10, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=15,
                                                    min_samples_split=5, random_state=42, n_jobs=-1),
    }

    for name, model in models.items():
        if name == "Linear Regression":
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2   = float(r2_score(y_test, y_pred))
        mae  = float(mean_absolute_error(y_test, y_pred))

        trained_models[name] = model
        model_metrics[name]  = {"RMSE": rmse, "R2": r2, "MAE": mae}

    # Feature importances from Random Forest
    rf = trained_models["Random Forest"]
    importances = rf.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    dataset_info["feature_importances"] = [{"name": n, "importance": round(float(v), 4)} for n, v in feat_imp[:15]]

    print("✅ All models trained and ready.")


# ── Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dashboard-data")
def dashboard_data():
    """Return all data needed by the frontend dashboard."""
    return jsonify({
        "dataset_info":   dataset_info,
        "model_metrics":  model_metrics,
        "feature_stats":  {k: v for k, v in feature_stats.items()
                           if k in ["living area", "lot area", "number of bedrooms",
                                    "number of bathrooms", "number of floors",
                                    "grade of the house", "condition of the house",
                                    "waterfront present", "number of views",
                                    "Built Year", "Renovation Year",
                                    "Number of schools nearby", "Distance from the airport"]},
        "feature_importances": dataset_info.get("feature_importances", []),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict price using all three models."""
    data = request.get_json()

    # Build feature vector with defaults
    row = {}
    defaults = {col: feature_stats[col]["median"] for col in feature_cols}
    for col in feature_cols:
        row[col] = float(data.get(col, defaults[col]))

    # Compute derived features if not provided
    if "house_age" not in data:
        row["house_age"] = 2026 - row.get("Built Year", 1975)
    if "is_renovated" not in data:
        row["is_renovated"] = 1 if row.get("Renovation Year", 0) > 0 else 0
    if "basement_ratio" not in data:
        row["basement_ratio"] = row.get("Area of the basement", 0) / (row.get("Area of the house(excluding basement)", 1580) + 1)
    if "total_area" not in data:
        row["total_area"] = row.get("Area of the house(excluding basement)", 1580) + row.get("Area of the basement", 0)
    if "bed_bath_ratio" not in data:
        row["bed_bath_ratio"] = row.get("number of bedrooms", 3) / (row.get("number of bathrooms", 2) + 1)

    X_input = pd.DataFrame([row])[feature_cols]

    predictions = {}
    for name, model in trained_models.items():
        if name == "Linear Regression":
            X_s = scaler.transform(X_input)
            pred = float(model.predict(X_s)[0])
        else:
            pred = float(model.predict(X_input)[0])
        predictions[name] = round(pred, 2)

    return jsonify({"predictions": predictions})


@app.route("/plots/<path:filename>")
def serve_plot(filename):
    return send_from_directory(PLOT_DIR, filename)


if __name__ == "__main__":
    train_all_models()
    app.run(debug=False, port=5050)
