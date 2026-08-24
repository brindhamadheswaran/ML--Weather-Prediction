# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Create the environmental sensor dataset containing Humidity, Pressure, Wind Speed, CO₂, Temperature, PM2.5, and Energy values.
2.Select the sensor parameters as input features and Temperature, PM2.5, and Energy as target variables.
3.Split the data into training and testing sets and create separate Random Forest Regression models for the three targets.
4.Train the models and predict Temperature, PM2.5, and Energy using the test data.
5.Evaluate and display the results using MSE and R² score, 
predict values for new sensor data, and 
plot actual versus predicted values.
```

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by:BRINDHA.M
RegisterNumber:  212225060038
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create sample dataset

data = {
    'Humidity': [60, 65, 55, 70, 62, 58, 75, 68, 52, 63],
    'Pressure': [1010, 1008, 1012, 1005, 1009, 1011, 1004, 1007, 1013, 1009],
    'WindSpeed': [12, 10, 14, 8, 11, 13, 7, 9, 15, 10],
    'CO2': [450, 480, 420, 510, 460, 440, 530, 500, 410, 470],

    'Temperature': [28.5, 29.1, 27.8, 30.2, 28.8,
                    28.2, 31.0, 29.8, 27.5, 29.0],

    'PM2.5': [35, 40, 30, 48, 37, 32, 52, 45, 28, 39],

    'Energy': [120, 125, 115, 135, 122,
               118, 140, 132, 112, 124]
}

df = pd.DataFrame(data)
# Step 2: Select input features

X = df[['Humidity', 'Pressure', 'WindSpeed', 'CO2']]

# Step 3: Select target variables
y_temperature = df['Temperature']
y_pm25 = df['PM2.5']
y_energy = df['Energy']

# Step 4: Split the data

X_train, X_test, y_temp_train, y_temp_test = train_test_split(
    X, y_temperature, test_size=0.2, random_state=42
)

_, _, y_pm25_train, y_pm25_test = train_test_split(
    X, y_pm25, test_size=0.2, random_state=42
)

_, _, y_energy_train, y_energy_test = train_test_split(
    X, y_energy, test_size=0.2, random_state=42
)
# Step 5: Create Random Forest models

temp_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

pm25_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

energy_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Step 6: Train the models

temp_model.fit(X_train, y_temp_train)
pm25_model.fit(X_train, y_pm25_train)
energy_model.fit(X_train, y_energy_train)

# Step 7: Make predictions

temp_pred = temp_model.predict(X_test)
pm25_pred = pm25_model.predict(X_test)
energy_pred = energy_model.predict(X_test)

# Step 8: Evaluate the models

print("Temperature Prediction")
print("MSE:", mean_squared_error(y_temp_test, temp_pred))
print("R2 Score:", r2_score(y_temp_test, temp_pred))

print("\nPM2.5 Prediction")
print("MSE:", mean_squared_error(y_pm25_test, pm25_pred))
print("R2 Score:", r2_score(y_pm25_test, pm25_pred))

print("\nEnergy Prediction")
print("MSE:", mean_squared_error(y_energy_test, energy_pred))
print("R2 Score:", r2_score(y_energy_test, energy_pred))


# Step 9: Display predictions

print("\nPredicted Temperature:", temp_pred)
print("Predicted PM2.5:", pm25_pred)
print("Predicted Energy:", energy_pred)

# Step 10: Predict new sensor data

new_data = pd.DataFrame({
    'Humidity': [60],
    'Pressure': [1010],
    'WindSpeed': [12],
    'CO2': [450]
})

temperature = temp_model.predict(new_data)
pm25 = pm25_model.predict(new_data)
energy = energy_model.predict(new_data)

print("\nNew Sensor Data Prediction")
print("Temperature:", temperature[0])
print("PM2.5:", pm25[0])
print("Energy:", energy[0])

# Step 11: Plot results

plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.scatter(y_temp_test, temp_pred)
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Temperature Prediction")

plt.subplot(1, 3, 2)
plt.scatter(y_pm25_test, pm25_pred)
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("PM2.5 Prediction")

plt.subplot(1, 3, 3)
plt.scatter(y_energy_test, energy_pred)
plt.xlabel("Actual Energy")
plt.ylabel("Predicted Energy")
plt.title("Energy Prediction")

plt.tight_layout()
plt.show()
```

## Output:
<img width="446" height="415" alt="image" src="https://github.com/user-attachments/assets/32dc47cb-55fb-499e-a6f1-7176a1f95b04" />

<img width="911" height="380" alt="image" src="https://github.com/user-attachments/assets/918e6636-3800-4e1a-866d-2c34bc9fccda" />

## Result:
