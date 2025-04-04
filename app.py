from flask import Flask, render_template, request
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import folium
from folium.plugins import HeatMap
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)

# Load dataset
df = pd.read_excel("Car Sales.xlsx", engine="openpyxl")

@app.route("/")
def home():
    return render_template("index.html")

# 1️⃣ Predict Car Price
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Load XGBoost model and column names
            model = joblib.load("car_price_xgboost.pkl")
            with open("model_columns.txt", "r") as f:
                model_columns = f.read().split(",")
            print("Model and columns loaded successfully!")
        except FileNotFoundError:
            error_message = "Model or column names not found. Please train the model first."
            print(error_message)
            return render_template("predict.html", predicted_price=None, error=error_message)
        
        # Get user input
        input_data = {
            "Model": request.form["model"],
            "Engine": request.form["engine"],
            "Transmission": request.form["transmission"],
            "Body Style": request.form["body_style"],
            "Annual Income": float(request.form["annual_income"]),
        }
        print("Input Data:", input_data)
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # One-hot encode the input data
        input_encoded = pd.get_dummies(input_df)
        print("One-Hot Encoded Input Data:", input_encoded)
        
        # Ensure the input data has the same columns as the training data
        # Exclude "Price ($)" from the model columns
        model_columns = [col for col in model_columns if col != "Price ($)"]
        
        # Add missing columns with default value 0
        missing_columns = set(model_columns) - set(input_encoded.columns)
        for column in missing_columns:
            input_encoded[column] = 0
        
        # Reorder columns to match the training data
        try:
            input_encoded = input_encoded[model_columns]
        except KeyError as e:
            print("Column mismatch error:", e)
            return render_template("predict.html", predicted_price=None, error="Column mismatch in input data. Please retrain the model.")
        
        print("Aligned Input Data Columns:", input_encoded.columns)
        
        # Make prediction
        try:
            predicted_price = model.predict(input_encoded)[0]
            print("Predicted Price:", predicted_price)
        except Exception as e:
            print("Prediction error:", e)
            return render_template("predict.html", predicted_price=None, error="Error during prediction. Check model and input format.")
        
        return render_template("predict.html", predicted_price=round(predicted_price, 2))
    
    return render_template("predict.html", predicted_price=None)

# 2️⃣ Customer Segmentation
@app.route("/customer-segmentation")
def customer_segmentation():
    features = ["Annual Income", "Price ($)", "Gender", "Dealer_Region", "Body Style", "Engine"]
    df_filtered = df[features].dropna()
    df_encoded = pd.get_dummies(df_filtered, columns=["Gender", "Dealer_Region", "Body Style", "Engine"], drop_first=True)
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_encoded)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_filtered["Cluster"] = kmeans.fit_predict(df_scaled)
    
    cluster_data = {}
    for cluster in range(3):
        cluster_df = df_filtered[df_filtered["Cluster"] == cluster]
        cluster_data[f"Segment {cluster + 1}"] = {
            "count": len(cluster_df),
            "attributes": {
                "Avg Annual Income": round(cluster_df["Annual Income"].mean(), 1),
                "Avg Car Price ($)": round(cluster_df["Price ($)"].mean(), 1),
                "Top Dealer Region": cluster_df["Dealer_Region"].mode()[0],
                "Top Body Style": cluster_df["Body Style"].mode()[0],
                "Top Engine Type": cluster_df["Engine"].mode()[0],
            }
        }
    return render_template("customer_segmentation.html", segments=cluster_data)

# 3️⃣ Regional Car Sales
@app.route("/regional-sales")
def regional_sales():
    # Group sales by region and calculate total sales
    sales_by_region = df.groupby("Dealer_Region").size().to_dict()
    
    # Pass the sales data to the template
    return render_template("regional_sales.html", sales=sales_by_region)

# 4️⃣ Most Sold Cars

    
@app.route("/top-selling")
def top_selling():
    # Calculate the top 5 most sold cars
    top_cars = df["Model"].value_counts().head(5).to_dict()
    
    # Pass the top cars to the template
    return render_template("top_selling.html", cars=top_cars)
# 5️⃣ Least Sold Cars
@app.route("/least-selling")
def least_selling():
    least_cars = df["Model"].value_counts().tail(5).to_dict()
    return render_template("least_selling.html", cars=least_cars)

if __name__ == "__main__":
    app.run(debug=True)
