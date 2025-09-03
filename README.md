# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Data Collection:
Import essential libraries such as pandas, numpy, sklearn, matplotlib, and seaborn.
Load the dataset using pandas.read_csv().
Data Preparation:

2.Address any missing values in the dataset.
Select key features for training the models.
Split the data into training and testing sets using train_test_split().
Linear Regression:

3.Initialize a Linear Regression model using sklearn.
Train the model on the training data with the .fit() method.
Use the model to predict values for the test set with .predict().
Evaluate performance using metrics like Mean Squared Error (MSE) and R² score.
Polynomial Regression:

4.Generate polynomial features using PolynomialFeatures from sklearn.
Train a Linear Regression model on the transformed polynomial dataset.
Make predictions and assess the model's performance similarly to the Linear Regression approach.
Visualization:

5.Plot regression lines for both Linear and Polynomial models.
Visualize residuals to analyze the models' performance.

## Program:
```
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: DHANAPPRIYA S
RegisterNumber:  212224230056
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("CarPrice_Assignment.csv")

X = df[['enginesize', 'horsepower', 'citympg', 'highwaympg']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


linear_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)


poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)


print("Name: S VASUNTHARA SAI")
print("Reg. No: 212224230297  ")
print("Linear Regression:")
print("MSE:", mean_squared_error(y_test, y_pred_linear))
print("R2 Score:", r2_score(y_test, y_pred_linear))

print(f"MSE: {mean_squared_error(y_test, y_pred_linear):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_linear):.2f}")

print("\nPolynomial Regression:")
print("MSE:", mean_squared_error(y_test, y_pred_poly))
print("R2 Score:", r2_score(y_test, y_pred_poly))

print(f"MSE: {mean_squared_error(y_test, y_pred_poly):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_poly):.2f}")


plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_linear, label='Linear', alpha=0.6)
plt.scatter(y_test, y_pred_poly, label='Polynomial (degree=2)', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', label='Perfect Prediction')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.legend()
plt.title("Linear vs Polynomial Regression")
plt.show()

```

## Output:
<img width="1101" height="810" alt="image" src="https://github.com/user-attachments/assets/37294920-3f80-47d0-87b7-53404596f8d7" />

## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.

