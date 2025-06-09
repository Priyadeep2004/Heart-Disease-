import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# --- IMPORTANT: Data Loading and Feature Preparation Section ---
# You NEED to replace this section with YOUR actual data loading and feature engineering.
#
# If your heart.csv contains physiological data (e.g., age, cholesterol, etc., and a 'target' column),
# uncomment the line below and ensure the path is correct.
# df = pd.read_csv('your_path_to_heart_disease_data.csv')
#
# If your goal is to use the 'heart.csv' with 'file_path' and 'predicted_label':
# You MUST first extract numerical features from the audio files specified by 'file_path'.
# This is a complex step involving audio processing libraries (e.g., librosa) and cannot be done directly here.
# After extraction, you would create a DataFrame with these numerical features.
#
# For demonstration purposes, I will generate a simple dummy dataset
# with numerical features and a target column.
# In your real application, load your actual dataset here.

print("--- Simulating a numerical dataset for demonstration ---")
print("Please replace this section with your actual data loading and feature engineering!")

# Create a dummy DataFrame with numerical features for demonstration
# In a real scenario, 'age', 'chol', etc., would come from your heart disease CSV.
num_samples = 100
data = {
    'age': np.random.randint(20, 80, num_samples),
    'sex': np.random.randint(0, 2, num_samples),
    'chol': np.random.randint(100, 300, num_samples),
    'max_hr': np.random.randint(100, 200, num_samples),
    'st_dep': np.round(np.random.uniform(0.0, 4.0, num_samples), 1),
    'num_vessels': np.random.randint(0, 4, num_samples),
    'target': np.random.randint(0, 2, num_samples) # 0 for no disease, 1 for disease
}
df = pd.DataFrame(data)

# Ensure column names are clean and lowercase
df.columns = df.columns.str.strip().str.lower()

# --- End of Data Loading and Feature Preparation Section ---


# Basic data inspection
print("\nDataFrame Head:")
print(df.head())
print("\nDataFrame Info:")
print(df.info())
print("\nDataFrame Description:")
print(df.describe())

# Remove duplicates if any
df = df.drop_duplicates()
print(f"\nDataFrame shape after dropping duplicates: {df.shape}")

# Check for null values
print("\nMissing values:\n", df.isnull().sum())

# Plot target distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='target')
plt.title("Target Class Distribution (Dummy Data)")
plt.xlabel("Target (0: No Disease, 1: Disease)")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Plot correlation heatmap
plt.figure(figsize=(10, 7))
# Calculate correlation only for numerical columns
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Feature Correlation (Dummy Data)")
plt.show()

# Prepare features and labels
# The 'target' column is assumed to be the label to predict.
X = df.drop('target', axis=1)
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify the shapes of the splits
print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Train model
print("\nTraining RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("\n--- Model Evaluation (Dummy Data) ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

# Define the filename for the model
model_filename = 'heart_model.pkl'

# Save model to file
try:
    joblib.dump(model, model_filename)
    print(f"\nModel successfully saved as {model_filename}")
    # Verify if the file exists
    if os.path.exists(model_filename):
        print(f"Confirmation: {model_filename} exists in the current directory.")
except Exception as e:
    print(f"Error saving model: {e}")

# Optional: Load the model to verify
try:
    loaded_model = joblib.load(model_filename)
    print(f"\nModel successfully loaded from {model_filename} for verification.")
    # You can perform a quick prediction with the loaded model
    # if X_test is not empty
    if not X_test.empty:
        sample_prediction = loaded_model.predict(X_test.head(1))
        print(f"Sample prediction with loaded model on first test sample: {sample_prediction}")
    else:
        print("X_test is empty, cannot perform sample prediction with loaded model.")
except Exception as e:
    print(f"Error loading model for verification: {e}")
