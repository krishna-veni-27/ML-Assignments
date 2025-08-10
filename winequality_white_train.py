import pandas as pd
import sklearn.neighbors as ng
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import math

# Load the dataset
mydata = pd.read_csv("winequality-white.csv")
x = mydata[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]
y = mydata["quality"].values  # Converting to 1D array

# Normalizing the features
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# Building the neural network model
wine_white_model = Sequential()
wine_white_model.add(Dense(64,activation = "relu",input_shape =(11,)))
wine_white_model.add(Dense(32,activation = "relu"))
wine_white_model.add(Dense(1))
wine_white_model.compile(optimizer = "adam",loss ="mse",metrics = ["mae"])
wine_white_model.fit(x_train,y_train,epochs = 50)
print("Training Completed......")

# Save the model & the scaler
wine_white_model.save("wine_white_model.keras")
joblib.dump(scaler,"scaler_white.pkl")

# Evaluate the model
test_result = wine_white_model.predict(x_test)
print("MSE",mean_squared_error(y_test,test_result))
print("RMSE",math.sqrt(mean_squared_error(y_test,test_result)))
print("MAE",mean_absolute_error(y_test,test_result))