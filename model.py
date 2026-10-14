import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("fraud_data.csv")

# Input features
X = data[
    ["amount", "transaction_hour", "location_risk", "previous_fraud"]
]

# Output
y = data["fraud"]

# Create and train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Save trained model
joblib.dump(model, "fraud_model.pkl")

print("Fraud detection model trained successfully!")