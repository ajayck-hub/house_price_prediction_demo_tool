# Import necessary libraries
import pandas as pd
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#Load the california Housing dataset
housing = fetch_california_housing(as_frame=True)

#Create a Dataframe from the dataset
df = housing.frame

print("California Housing Data: ")
print(df.head())

# Features(Independent variables) and target(Dependent variable)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

#Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

#Evaluate the model using MSE and R2 Score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

print("Model Coefficiens: ")
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])

print("Coefficient for each feature")
print(coef_df)

#Test the model with new data
new_data = pd.DataFrame({
    'MedInc':[5],
    'HouseAge':[30],
    'AveRooms':[6],
    'AveBedrms':[1],
    'Population':[500],
    'AveOccup': [30],
    'Latitude':[34.05],
    'Longitude':[-118.25]
})
predicted_price = model.predict(new_data)
print(f"\n\n Predicted House Price: ${predicted_price[0]:,.2f}")



