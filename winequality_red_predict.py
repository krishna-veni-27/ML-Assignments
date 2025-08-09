import joblib
import numpy as np

wine_model = joblib.load("wine_model.pkl")
scaler = joblib.load("scaler.pkl")

input_data = np.array([[6.9,0.4,0.14,2.4,0.085,21.0,40.0,0.9968,3.43,0.63,9.7]]) 
scaled_input = scaler.transform(input_data)

Predicted_Quality = wine_model.predict(scaled_input)
print("Predicted_Quality:",Predicted_Quality[0][0])
Rounded_Quality = round(Predicted_Quality[0][0])
print("Rounded_Quality:",Rounded_Quality)

