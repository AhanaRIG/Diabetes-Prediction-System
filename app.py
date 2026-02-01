from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "RandomForest.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            input_values = [
                int(request.form["Pregnancies"]),
                float(request.form["Glucose"]),
                float(request.form["BloodPressure"]),
                float(request.form["SkinThickness"]),
                float(request.form["Insulin"]),
                float(request.form["BMI"]),
                float(request.form["DiabetesPedigreeFunction"]),
                int(request.form["Age"])
            ]

            input_array = np.array([input_values])
            input_scaled = scaler.transform(input_array)
            result = model.predict(input_scaled)[0]

            prediction = "Diabetic" if result == 1 else "Not Diabetic"
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
