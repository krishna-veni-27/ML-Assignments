from tensorflow.keras.models import load_model
import joblib
import numpy as np

wine_red_model = load_model("wine_red_model.keras")
scaler = joblib.load("scaler_red.pkl")

fa=float(input("enter fixed acidity:"))
va=float(input("enter volatile acidity:"))
ca=float(input("enter citric acid:"))
rs=float(input("enter residual sugar:"))
c=float(input("enter chlorides:"))
fsd=float(input("enter free sulphur dioxide:"))
tsd=float(input("enter total sulphur dioxide:"))
d=float(input("enter density:"))
ph=float(input("enter ph:"))
s=float(input("enter sulphates:"))
a=float(input("enter alcohol:"))

input_data = np.array([[fa,va,ca,rs,c,fsd,tsd,d,ph,s,a]])
scaled_input = scaler.transform(input_data)

Predicted_Quality = wine_red_model.predict(scaled_input)
print("Predicted_Quality:",Predicted_Quality[0][0])
Rounded_Quality = round(Predicted_Quality[0][0])
print("Rounded_Quality:",Rounded_Quality)
Accurate_Quality = int(Predicted_Quality[0][0])
print("Accurate_Quality:",Accurate_Quality)



