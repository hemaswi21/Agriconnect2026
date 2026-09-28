from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

data = pd.read_csv("dataset model/Crop_recommendation.csv")

X = data.drop("label", axis=1)
y = data["label"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)


@app.route("/", methods=["GET", "POST"])
def home():
    result = ""

    if request.method == "POST":
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        input_data = pd.DataFrame([[
            N, P, K, temperature, humidity, ph, rainfall
        ]], columns=X.columns)

        result = model.predict(input_data)[0]

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)