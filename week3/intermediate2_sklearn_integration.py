"""
Implementing NumPy Fundamentals using Scikit-learn
Integration of NumPy arrays with machine learning workflows
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def demonstrate_numpy_sklearn_integration():
    """
    Demonstrate how NumPy fundamentals integrate with Scikit-learn
    """
    print("=== NumPy Fundamentals with Scikit-learn ===\n")
    
    # 1. Data Generation and Preparation
    print("1. DATA GENERATION AND PREPARATION")
    
    # Generate synthetic dataset using NumPy-compatible arrays
    X, y = make_classification(
        n_samples=1000, 
        n_features=4, 
        n_informative=3, 
        n_redundant=1,
        n_clusters_per_class=1,
        random_state=42
    )
    
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Target vector shape: {y.shape}")
    print(f"   Feature matrix type: {type(X)}")  # numpy.ndarray
    print(f"   Unique classes: {np.unique(y)}")
    
    # 2. Data Splitting with NumPy arrays
    print("\n2. DATA SPLITTING")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training set: {X_train.shape}, {y_train.shape}")
    print(f"   Test set: {X_test.shape}, {y_test.shape}")
    
    # 3. Feature Scaling using NumPy arrays
    print("\n3. FEATURE SCALING")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Returns NumPy array
    X_test_scaled = scaler.transform(X_test)        # Returns NumPy array
    
    print(f"   Scaled training data - Mean: {np.mean(X_train_scaled, axis=0)}")
    print(f"   Scaled training data - Std: {np.std(X_train_scaled, axis=0)}")
    
    # 4. Feature Engineering with Polynomial Features
    print("\n4. FEATURE ENGINEERING")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    
    print(f"   Original features: {X_train_scaled.shape[1]}")
    print(f"   Polynomial features: {X_train_poly.shape[1]}")
    
    return X_train_scaled, X_test_scaled, X_train_poly, X_test_poly, y_train, y_test

def build_classification_models(X_train, X_test, X_train_poly, X_test_poly, y_train, y_test):
    """
    Build and compare classification models
    """
    print("\n5. CLASSIFICATION MODELS")
    
    # Model 1: Logistic Regression with original features
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred_log_reg = log_reg.predict(X_test)
    accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
    
    # Model 2: Logistic Regression with polynomial features
    log_reg_poly = LogisticRegression(random_state=42, max_iter=1000)
    log_reg_poly.fit(X_train_poly, y_train)
    y_pred_log_reg_poly = log_reg_poly.predict(X_test_poly)
    accuracy_log_reg_poly = accuracy_score(y_test, y_pred_log_reg_poly)
    
    # Model 3: Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    
    print(f"   Logistic Regression Accuracy: {accuracy_log_reg:.4f}")
    print(f"   Logistic Regression (Poly) Accuracy: {accuracy_log_reg_poly:.4f}")
    print(f"   Random Forest Accuracy: {accuracy_rf:.4f}")
    
    return {
        'log_reg': (log_reg, y_pred_log_reg, accuracy_log_reg),
        'log_reg_poly': (log_reg_poly, y_pred_log_reg_poly, accuracy_log_reg_poly),
        'random_forest': (rf, y_pred_rf, accuracy_rf)
    }

def dimensionality_reduction_demo(X_train, X_test, y_train, y_test):
    """
    Demonstrate dimensionality reduction with PCA
    """
    print("\n6. DIMENSIONALITY REDUCTION WITH PCA")
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(f"   Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"   Total explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Train model on reduced features
    log_reg_pca = LogisticRegression(random_state=42)
    log_reg_pca.fit(X_train_pca, y_train)
    y_pred_pca = log_reg_pca.predict(X_test_pca)
    accuracy_pca = accuracy_score(y_test, y_pred_pca)
    
    print(f"   Logistic Regression with PCA Accuracy: {accuracy_pca:.4f}")
    
    return X_train_pca, X_test_pca, pca

def regression_demo():
    """
    Demonstrate regression with NumPy and Scikit-learn
    """
    print("\n7. REGRESSION DEMONSTRATION")
    
    # Generate regression dataset
    X_reg, y_reg = make_regression(
        n_samples=500, 
        n_features=3, 
        noise=10, 
        random_state=42
    )
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler_reg = StandardScaler()
    X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler_reg.transform(X_test_reg)
    
    # Train regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_reg_scaled, y_train_reg)
    y_pred_reg = lin_reg.predict(X_test_reg_scaled)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    
    print(f"   Linear Regression MSE: {mse:.4f}")
    print(f"   R² Score: {lin_reg.score(X_test_reg_scaled, y_test_reg):.4f}")
    print(f"   Coefficients: {lin_reg.coef_}")
    print(f"   Intercept: {lin_reg.intercept_:.4f}")
    
    return lin_reg, X_test_reg_scaled, y_test_reg, y_pred_reg

def visualize_results(X_train_pca, y_train, models, X_test, y_test):
    """
    Create visualizations of the results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. PCA visualization
    scatter = axes[0, 0].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, 
                                cmap='viridis', alpha=0.7)
    axes[0, 0].set_title('PCA: First Two Principal Components')
    axes[0, 0].set_xlabel('First Principal Component')
    axes[0, 0].set_ylabel('Second Principal Component')
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # 2. Model comparison
    model_names = ['Logistic Reg', 'Logistic Reg (Poly)', 'Random Forest']
    accuracies = [models['log_reg'][2], models['log_reg_poly'][2], models['random_forest'][2]]
    
    bars = axes[0, 1].bar(model_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 1].set_title('Model Accuracy Comparison')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{accuracy:.4f}', ha='center', va='bottom')
    
    # 3. Feature importance from Random Forest
    feature_importance = models['random_forest'][0].feature_importances_
    features = [f'Feature {i+1}' for i in range(len(feature_importance))]
    
    axes[1, 0].barh(features, feature_importance, color='lightseagreen')
    axes[1, 0].set_title('Random Forest Feature Importance')
    axes[1, 0].set_xlabel('Importance')
    
    # 4. Confusion Matrix for best model
    from sklearn.metrics import confusion_matrix
    best_model_pred = models['random_forest'][1]
    cm = confusion_matrix(y_test, best_model_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_title('Confusion Matrix - Random Forest')
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('sklearn_numpy_integration.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function demonstrating NumPy-Scikit-learn integration
    """
    # Demonstrate integration
    X_train, X_test, X_train_poly, X_test_poly, y_train, y_test = demonstrate_numpy_sklearn_integration()
    
    # Build classification models
    models = build_classification_models(X_train, X_test, X_train_poly, X_test_poly, y_train, y_test)
    
    # Dimensionality reduction demo
    X_train_pca, X_test_pca, pca = dimensionality_reduction_demo(X_train, X_test, y_train, y_test)
    
    # Regression demo
    lin_reg, X_test_reg, y_test_reg, y_pred_reg = regression_demo()
    
    # Visualize results
    visualize_results(X_train_pca, y_train, models, X_test, y_test)
    
    print("\n=== Key Takeaways ===")
    print("1. Scikit-learn seamlessly works with NumPy arrays")
    print("2. All preprocessing (scaling, feature engineering) returns NumPy arrays")
    print("3. Model training and prediction use NumPy's efficient operations")
    print("4. NumPy enables easy manipulation and analysis of model outputs")
    print("5. Integration allows for efficient machine learning workflows")

if __name__ == "__main__":
    main()