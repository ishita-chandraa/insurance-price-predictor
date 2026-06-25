import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("insurance.csv")

# Features and target
X = data.drop("charges", axis=1)
y = data["charges"]

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Polynomial Features
poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

print("Original Shape:", X.shape)
print("Polynomial Shape:", X_poly.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_poly,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)

print("\nPolynomial Regression MAE:", mae)

# Save model
joblib.dump(model, "insurance_polynomial_model.pkl")

print("Number of polynomial features:",
      X_poly.shape[1])

rmse = root_mean_squared_error(y_test, predictions)

print("RMSE:", rmse)

r2 = r2_score(y_test, predictions)

print("R² Score:", r2)
joblib.dump(model, "insurance_polynomial_model.pkl")
joblib.dump(poly, "poly_transformer.pkl")
print("\nPolynomial Model Saved!")