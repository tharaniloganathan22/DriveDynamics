import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
df = pd.read_excel("Car Sales.xlsx", engine="openpyxl")

# Select relevant features for prediction
features = ["Model", "Engine", "Transmission", "Body Style", "Annual Income"]
df_filtered = df[features + ["Price ($)"]].dropna()

# Encode categorical variables
df_encoded = pd.get_dummies(df_filtered, columns=["Model", "Engine", "Transmission", "Body Style"])

# Save the column names
with open("model_columns.txt", "w") as f:
    f.write(",".join(df_encoded.columns))

# Split data into features (X) and target (y)
X = df_encoded.drop("Price ($)", axis=1)
y = df_encoded["Price ($)"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))

# Save model to a file
joblib.dump(model, "car_price_xgboost.pkl")
print("Model saved to car_price_xgboost.pkl")