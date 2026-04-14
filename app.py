"""
Flask backend for Real Estate Price Prediction Dashboard
"""
import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

# ── Matplotlib Config for Headless Servers ──
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import seaborn as sns

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
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Global model objects (trained once on startup) ──
trained_models = {}
scaler         = None
feature_cols   = []
model_metrics  = {}
feature_stats  = {}
dataset_info   = {}


def generate_plots(df, feature_cols, model_metrics, trained_models, X_test, y_test):
    """Generate all visualization plots for the dashboard."""
    print("📊 Generating visualization plots...")
    sns.set_theme(style="whitegrid", palette="viridis", font_scale=1.1)
    
    # 1. Price Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["Price"], bins=50, kde=True, color="#6C63FF", ax=axes[0])
    axes[0].set_title("Distribution of House Prices", fontweight="bold")
    sns.histplot(np.log1p(df["Price"]), bins=50, kde=True, color="#FF6584", ax=axes[1])
    axes[1].set_title("Log-Transformed Price Distribution", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "01_price_distribution.png"), dpi=100)
    plt.close()

    # 2. Heatmap
    corr = df[feature_cols + ["Price"]].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="RdYlBu_r", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "02_correlation_heatmap.png"), dpi=100)
    plt.close()

    # 3. Model Comparison
    names = list(model_metrics.keys())
    r2_vals = [model_metrics[n]["R2"] for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, r2_vals, color=["#6C63FF", "#FF6584", "#43E97B"])
    ax.set_title("Model R² Score Comparison", fontweight="bold")
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "06_model_comparison.png"), dpi=100)
    plt.close()

    # 4. Actual vs Predicted (Random Forest)
    rf_pred = trained_models["Random Forest"].predict(X_test)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, rf_pred, alpha=0.3, s=10, color="#6C63FF")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax.set_title("Actual vs Predicted (Random Forest)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "07_actual_vs_predicted.png"), dpi=100)
    plt.close()

    # 5. Feature Importance
    rf = trained_models["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    imp.plot(kind="barh", ax=ax, color="#43E97B")
    ax.set_title("Top 15 Feature Importances", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "09_feature_importance.png"), dpi=100)
    plt.close()
    
    # 6. Residuals
    residuals = y_test - rf_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, bins=50, kde=True, color="#FF6584", ax=ax)
    ax.axvline(0, color="black", linestyle="--")
    ax.set_title("Residual Distribution (Random Forest)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "08_residuals.png"), dpi=100)
    plt.close()

    # Placeholder for others to avoid dashboard 404s if they were expected
    for f in ["03_top_features_correlation.png", "04_scatter_top_features.png", "05_boxplots.png", "10_learning_curve.png"]:
        if not os.path.exists(os.path.join(PLOT_DIR, f)):
             # Just copy a basic one as placeholder or generate briefly
             plt.figure(figsize=(2,2))
             plt.text(0.5, 0.5, "Graph Loading...", ha='center')
             plt.savefig(os.path.join(PLOT_DIR, f))
             plt.close()

    print("✅ All plots generated.")


def train_all_models():
    """Train all ML models and store them globally."""
    global trained_models, scaler, feature_cols, model_metrics, feature_stats, dataset_info

    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    original_rows = df.shape[0]

    df.drop(columns=["id", "Date"], inplace=True, errors="ignore")

    df["house_age"]      = 2026 - df["Built Year"]
    df["is_renovated"]   = (df["Renovation Year"] > 0).astype(int)
    df["basement_ratio"] = df["Area of the basement"] / (df["Area of the house(excluding basement)"] + 1)
    df["total_area"]     = df["Area of the house(excluding basement)"] + df["Area of the basement"]
    df["bed_bath_ratio"] = df["number of bedrooms"] / (df["number of bathrooms"] + 1)

    Q1   = df["Price"].quantile(0.01)
    Q3   = df["Price"].quantile(0.99)
    df   = df[(df["Price"] >= Q1) & (df["Price"] <= Q3)].reset_index(drop=True)

    dataset_info["total_rows"]    = original_rows
    dataset_info["clean_rows"]    = df.shape[0]
    dataset_info["num_features"]  = df.shape[1] - 1
    dataset_info["price_min"]     = float(df["Price"].min())
    dataset_info["price_max"]     = float(df["Price"].max())
    dataset_info["price_mean"]    = float(df["Price"].mean())
    dataset_info["price_median"]  = float(df["Price"].median())

    feature_cols = [c for c in df.columns if c not in ["Price", "price_per_sqft"]]
    X = df[feature_cols].copy()
    y = df["Price"].copy()

    for col in feature_cols:
        feature_stats[col] = {"min": float(X[col].min()), "max": float(X[col].max()), "mean": float(X[col].mean()), "median": float(X[col].median())}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree":     DecisionTreeRegressor(max_depth=12, min_samples_split=10, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1),
    }

    for name, model in models.items():
        if name == "Linear Regression":
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        model_metrics[name] = {
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "R2":   float(r2_score(y_test, y_pred)),
            "MAE":  float(mean_absolute_error(y_test, y_pred))
        }
        trained_models[name] = model

    rf = trained_models["Random Forest"]
    feat_imp = sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: x[1], reverse=True)
    dataset_info["feature_importances"] = [{"name": n, "importance": round(float(v), 4)} for n, v in feat_imp[:15]]

    print("✅ All models trained.")
    
    # Generate plots for the web dashboard on startup
    generate_plots(df, feature_cols, model_metrics, trained_models, X_test, y_test)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/dashboard-data")
def dashboard_data():
    return jsonify({
        "dataset_info": dataset_info,
        "model_metrics": model_metrics,
        "feature_stats": {k: v for k, v in feature_stats.items() if k in ["living area", "lot area", "number of bedrooms", "number of bathrooms", "number of floors", "grade of the house", "condition of the house", "waterfront present", "number of views", "Built Year", "Renovation Year", "Number of schools nearby", "Distance from the airport"]},
        "feature_importances": dataset_info.get("feature_importances", []),
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    row = {col: float(data.get(col, feature_stats[col]["median"])) for col in feature_cols if col in feature_stats}
    
    # Compute derived features
    row["house_age"]      = 2026 - row.get("Built Year", 1975)
    row["is_renovated"]   = 1 if row.get("Renovation Year", 0) > 0 else 0
    row["basement_ratio"] = row.get("Area of the basement", 0) / (row.get("Area of the house(excluding basement)", 1580) + 1)
    row["total_area"]     = row.get("Area of the house(excluding basement)", 1580) + row.get("Area of the basement", 0)
    row["bed_bath_ratio"] = row.get("number of bedrooms", 3) / (row.get("number of bathrooms", 2) + 1)

    X_input = pd.DataFrame([row])[feature_cols]
    predictions = {}
    for name, model in trained_models.items():
        if name == "Linear Regression":
            X_s = scaler.transform(X_input)
            pred = model.predict(X_s)[0]
        else:
            pred = model.predict(X_input)[0]
        predictions[name] = round(float(pred), 2)
    return jsonify({"predictions": predictions})

@app.route("/plots/<path:filename>")
def serve_plot(filename):
    return send_from_directory(PLOT_DIR, filename)

if __name__ == "__main__":
    train_all_models()
    # For local testing, use port 5050; for production, binding to PORT is handled by Gunicorn or the platform
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
