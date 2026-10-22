import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# Load the trained SVM model
model = joblib.load("svm.pkl")

# HTML template for a simple frontend form
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVM Model Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f7f6; margin: 0; padding: 20px; }
        .container { max-width: 600px; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); margin: auto; }
        h2 { color: #333; text-align: center; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
        input, select { width: 100%; padding: 8px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
        button { background-color: #007BFF; color: white; border: none; padding: 10px 15px; width: 100%; border-radius: 4px; font-size: 16px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        .result { margin-top: 20px; padding: 15px; background: #e9ecef; border-radius: 4px; text-align: center; font-size: 18px; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Health Risk Prediction (SVM)</h2>
        <form method="POST" action="/predict">
            <div class="form-group">
                <label>Age:</label>
                <input type="number" step="any" name="age" required>
            </div>
            <div class="form-group">
                <label>BMI:</label>
                <input type="number" step="any" name="bmi" required>
            </div>
            <div class="form-group">
                <label>Fasting Blood Sugar:</label>
                <input type="number" step="any" name="fasting_blood_sugar" required>
            </div>
            <div class="form-group">
                <label>HbA1c Level:</label>
                <input type="number" step="any" name="hba1c_level" required>
            </div>
            <div class="form-group">
                <label>Systolic Blood Pressure:</label>
                <input type="number" step="any" name="blood_pressure_systolic" required>
            </div>
            <div class="form-group">
                <label>Diastolic Blood Pressure:</label>
                <input type="number" step="any" name="blood_pressure_diastolic" required>
            </div>
            <div class="form-group">
                <label>Hours of Sleep per Night:</label>
                <input type="number" step="any" name="hours_sleep_per_night" required>
            </div>
            <div class="form-group">
                <label>Stress Level (1-10):</label>
                <input type="number" step="any" name="stress_level" required>
            </div>
            <button type="submit">Predict</button>
        </form>
        {% if prediction is not none %}
        <div class="result">
            Prediction Result: {{ prediction }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/")
def home():
  return render_template_string(HTML_TEMPLATE, prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
  try:
    # Extract values from form input and convert to floats
    # Note: Ensure this matches the exact feature count and order your model expects.
    form_data = [
        float(request.form.get("age", 0)),
        float(request.form.get("bmi", 0)),
        float(request.form.get("fasting_blood_sugar", 0)),
        float(request.form.get("hba1c_level", 0)),
        float(request.form.get("blood_pressure_systolic", 0)),
        float(request.form.get("blood_pressure_diastolic", 0)),
        float(request.form.get("hours_sleep_per_night", 0)),
        float(request.form.get("stress_level", 0)),
        # Add padding zeroes if your model requires more features than provided in this quick form template
    ]

    # If your model expects more features than included in this basic form UI,
    # pad the remaining slots with zeros or default averages:
    expected_features = getattr(model, "n_features_in_", len(form_data))
    if len(form_data) < expected_features:
      form_data.extend([0.0] * (expected_features - len(form_data)))

    final_input = np.array([form_data])
    prediction = model.predict(final_input)[0]

    return render_template_string(
        HTML_TEMPLATE, prediction=str(prediction)
    )
  except Exception as e:
    return render_template_string(
        HTML_TEMPLATE, prediction=f"Error processing input: {str(e)}"
    )


if __name__ == "__main__":
  app.run(debug=True)
