import pandas as pd
from pmdarima import auto_arima

# Example: Load your data
data = pd.read_csv('Car Sales.xlsx')  # Modify according to your dataset
X = data['feature_column'].values  # Modify based on your data structure
y = data['target_column'].values  # Modify based on your data structure

# Train the model
model = auto_arima(X, y, seasonal=True, m=12)

# Save the model (you can pickle or save it in another format)
import pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model training and saving completed.")
