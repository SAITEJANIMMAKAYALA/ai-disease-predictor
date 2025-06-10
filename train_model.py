import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the CSV data
df = pd.read_csv("symptom_disease_data.csv")

# Split features and target
X = df.drop("Disease", axis=1)
y = df["Disease"]

# Encode disease labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model and encoders
joblib.dump(model, "disease_model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(X.columns.tolist(), "symptoms_list.pkl")

print("✅ Model training complete. Files saved!")
