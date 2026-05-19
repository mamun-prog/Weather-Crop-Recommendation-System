import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/crop_data.csv")

# Features (inputs)
X = df.drop("label", axis=1)

# Target (output)
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

# Example prediction
sample_data = [[90, 42, 43, 20.8, 82, 6.5, 202]]

prediction = model.predict(sample_data)

print("\nRecommended Crop:")
print(prediction[0])