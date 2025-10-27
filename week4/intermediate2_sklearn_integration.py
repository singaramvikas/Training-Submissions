"""
Intermediate 2: Implement Pandas Fundamentals using appropriate library (Scikit-learn)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer

def pandas_sklearn_integration():
    """
    Demonstrate how Pandas integrates with Scikit-learn for machine learning
    """
    print("PANDAS + SCIKIT-LEARN INTEGRATION")
    print("=" * 50)
    
    # Create a more realistic dataset
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.normal(45, 15, n_samples).clip(18, 80),
        'income': np.random.normal(50000, 20000, n_samples).clip(20000, 150000),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                    n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        'credit_score': np.random.normal(650, 100, n_samples).clip(300, 850),
        'employment_years': np.random.exponential(5, n_samples).clip(0, 40),
        'loan_amount': np.random.normal(20000, 15000, n_samples).clip(1000, 100000),
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable (loan approval) based on features
    approval_prob = (
        (df['age'] > 25).astype(int) * 0.1 +
        (df['income'] > 40000).astype(int) * 0.3 +
        (df['education'].map({'High School': 0.1, 'Bachelor': 0.3, 'Master': 0.5, 'PhD': 0.7})) +
        (df['credit_score'] > 650).astype(int) * 0.2 +
        (df['employment_years'] > 2).astype(int) * 0.1
    )
    
    df['loan_approved'] = (approval_prob + np.random.normal(0, 0.2, n_samples) > 0.5).astype(int)
    
    # Add some missing values
    mask = np.random.random(df.shape) < 0.05  # 5% missing values
    df = df.mask(mask)
    
    print("1. Original Dataset:")
    print(f"Shape: {df.shape}")
    print(f"Loan Approval Rate: {df['loan_approved'].mean():.2%}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print()
    
    # 2. Data Preprocessing with Pandas
    print("2. Data Preprocessing:")
    
    # Handle categorical variables
    le = LabelEncoder()
    df['education_encoded'] = le.fit_transform(df['education'])
    
    # Feature selection
    features = ['age', 'income', 'education_encoded', 'credit_score', 'employment_years', 'loan_amount']
    X = df[features]
    y = df['loan_approved']
    
    print(f"Features: {features}")
    print(f"Target distribution:\n{y.value_counts()}")
    print()
    
    # 3. Handle missing values with Scikit-learn
    print("3. Handling Missing Values:")
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=features)
    
    print("Missing values after imputation:")
    print(X_imputed.isnull().sum())
    print()
    
    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("4. Data Splitting:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print()
    
    # 5. Feature scaling
    print("5. Feature Scaling:")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Feature statistics after scaling:")
    print(f"Mean: {X_train_scaled.mean(axis=0).round(2)}")
    print(f"Std: {X_train_scaled.std(axis=0).round(2)}")
    print()
    
    # 6. Model training
    print("6. Model Training:")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 7. Model evaluation
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print()
    
    # 8. Feature importance
    print("7. Feature Importance:")
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance)
    
    # 9. Create prediction pipeline function
    def predict_loan_approval(age, income, education, credit_score, employment_years, loan_amount):
        """Pipeline to predict loan approval for new data"""
        # Create input DataFrame
        input_data = pd.DataFrame({
            'age': [age],
            'income': [income],
            'education': [education],
            'credit_score': [credit_score],
            'employment_years': [employment_years],
            'loan_amount': [loan_amount]
        })
        
        # Preprocess
        input_data['education_encoded'] = le.transform(input_data['education'])
        input_features = input_data[features]
        
        # Handle potential missing values
        input_imputed = imputer.transform(input_features)
        
        # Scale features
        input_scaled = scaler.transform(input_imputed)
        
        # Predict
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)
        
        return prediction[0], probability[0][1]
    
    print("\n8. Prediction Pipeline Test:")
    test_case = (35, 60000, 'Bachelor', 720, 5, 25000)
    pred, prob = predict_loan_approval(*test_case)
    print(f"Test case: {test_case}")
    print(f"Prediction: {'Approved' if pred == 1 else 'Denied'}")
    print(f"Probability: {prob:.2%}")

if __name__ == "__main__":
    pandas_sklearn_integration()