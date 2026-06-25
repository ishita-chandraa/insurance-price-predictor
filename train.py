import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

data = pd.read_csv("insurance.csv")

print(data.head())
print(data.info())
X = data.drop("charges", axis=1)
y = data["charges"]

X = pd.get_dummies(
    X,
    drop_first=True
)

print(X.head())
print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("MAE:", mae)

rmse = root_mean_squared_error(y_test, predictions)

print("RMSE:", rmse)

r2 = r2_score(y_test, predictions)

print("R² Score:", r2)