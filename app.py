from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("fraud_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    amount = float(request.form["amount"])
    hour = int(request.form["hour"])
    location_risk = int(request.form["location_risk"])
    previous_fraud = int(request.form["previous_fraud"])

    transaction = pd.DataFrame(
        [[amount, hour, location_risk, previous_fraud]],
        columns=[
            "amount",
            "transaction_hour",
            "location_risk",
            "previous_fraud"
        ]
    )

    # Probability that transaction is fraudulent
    probability = model.predict_proba(transaction)[0][1]

    # Convert probability to 0-100 risk score
    risk_score = round(probability * 100, 2)

    if risk_score < 30:
        result = "Low Risk"
    elif risk_score < 70:
        result = "Medium Risk"
    else:
        result = "High Risk - Possible Fraud"

    return render_template(
        "result.html",
        result=result,
        risk_score=risk_score
    )


if __name__ == "__main__":
    app.run(debug=True)