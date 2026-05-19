import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv(
    r"C:\Users\HP\OneDrive\Desktop\Weather-Crop-Recommendation-System\data\crop_data.csv"
)

# Features and target
X = df.drop("label", axis=1)
y = df["label"]

# Train model
model = RandomForestClassifier()

model.fit(X, y)

# Save model
joblib.dump(model, "models/crop_model.pkl")

print("Model saved successfully!")