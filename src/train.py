import dagshub
import mlflow
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 📖 Connect MLflow tracking to your DagsHub repo
dagshub.init(
    repo_owner='santosh.flyingmachine',
    repo_name='dagshub_gitbot',
    mlflow=True
)

# 📚 Load dataset (semicolon-separated CSV from UCI Wine Quality dataset)
data = pd.read_csv("data/winequality-red.csv", sep=";")

# Features (X) and target (y)
X = data.drop("quality", axis=1)
y = (data["quality"] >= 6).astype(int)  # Good wine (>=6) vs Bad wine (<6)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 📖 Start a new MLflow run (experiment log)
with mlflow.start_run():
    # Parameters (edition of the book studied)
    C = 0.1
    max_iter = 1000  # avoid convergence warnings

    # Train model
    model = LogisticRegression(C=C, max_iter=max_iter)
    model.fit(X_train, y_train)

    # Predictions + accuracy
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Log parameters & metrics
    mlflow.log_param("C", C)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_metric("accuracy", acc)

    # Save model locally
    model_path = "models/model.pkl"
    joblib.dump(model, model_path)

    # Log model file as an artifact (supported in DagsHub MLflow)
    mlflow.log_artifact(model_path)

    # Save metrics for DVC tracking
    with open("metrics.json", "w") as f:
        json.dump({"accuracy": acc}, f)

    print(f"✅ Run logged to MLflow with accuracy: {acc:.4f}")
    print(f"📦 Model saved at: {model_path}")
