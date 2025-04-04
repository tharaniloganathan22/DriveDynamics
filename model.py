# model.py
import pandas as pd
import numpy as np
import pickle
from pmdarima import auto_arima

# Load real dataset
DATA_FILE = "car_sales.csv"  # Ensure this file is available

def train_and_save_model():
    data = pd.read_csv(DATA_FILE, parse_dates=['Date'])
    data = data.sort_values(by='Date')  # Ensure data is in chronological order
    y = data['Price ($)'].values  # Target variable: Car Price
    model = auto_arima(y, seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True)
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

# Train and save the model if not found
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    train_and_save_model()

def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def predict_future(n_periods):
    model = load_model()
    prediction = model.predict(n_periods=int(n_periods))  # Ensure n_periods is an integer
    return prediction.tolist()  # Return a list of predicted values