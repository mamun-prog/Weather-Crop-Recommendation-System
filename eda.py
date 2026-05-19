import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/crop_data.csv")

# Show first 5 rows
print(df.head())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Crop counts
print("\nCrop Counts:")
print(df['label'].value_counts())

# Plot temperature distribution
plt.figure(figsize=(8,5))
plt.hist(df['temperature'], bins=20)

plt.title("Temperature Distribution")
plt.xlabel("Temperature")
plt.ylabel("Frequency")

plt.show()