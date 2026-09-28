import pandas as pd

data = pd.read_csv("dataset model/Crop_recommendation.csv")

print(data.head())
print(data.shape)
print(data.columns)

import pandas as pd

data = pd.read_csv("dataset model/Crop_recommendation.csv")

print(data.head())
print(data.shape)
print(data.columns)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = data.drop("label", axis=1)
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

N = float(input("Enter N: "))
P = float(input("Enter P: "))
K = float(input("Enter K: "))
temperature = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
ph = float(input("Enter pH: "))
rainfall = float(input("Enter Rainfall: "))

user_input = pd.DataFrame([[
    N, P, K, temperature, humidity, ph, rainfall
]], columns=X.columns)

result = model.predict(user_input)

print("Recommended Crop:", result[0])