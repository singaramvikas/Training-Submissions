from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

def sklearn_basics_demo():
    """Demonstrate Python basics with Scikit-learn for machine learning"""
    
    # Load breast cancer dataset
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    
    print("=== Scikit-learn Basics Implementation ===")
    print(f"Dataset: {cancer.DESCR[:500]}...")
    print(f"Features: {cancer.feature_names}")
    print(f"Target: {cancer.target_names}")
    print(f"Data shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Basic data preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Model training
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Predictions and evaluation
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n=== Model Performance ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=cancer.target_names))
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': cancer.feature_names,
        'importance': abs(model.coef_[0])
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return model, scaler, accuracy

# Execute the implementation
model, scaler, accuracy = sklearn_basics_demo()