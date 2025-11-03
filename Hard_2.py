import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

class CustomerChurnPredictor:
    """
    End-to-end mini project for predicting customer churn
    Demonstrates complete Python basics & setup workflow
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        
    def generate_synthetic_data(self, n_samples=10000):
        """Generate synthetic customer churn data"""
        np.random.seed(42)
        
        data = {
            'age': np.random.randint(18, 80, n_samples),
            'monthly_charges': np.random.normal(64.76, 30.09, n_samples),
            'total_charges': np.random.normal(2283.30, 2266.77, n_samples),
            'tenure': np.random.randint(1, 72, n_samples),
            'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'monthly_charges_bin': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic churn patterns
        churn_proba = (
            (df['contract'] == 'Month-to-month') * 0.3 +
            (df['internet_service'] == 'Fiber optic') * 0.2 +
            (df['online_security'] == 'No') * 0.15 +
            (df['tech_support'] == 'No') * 0.1 +
            (df['monthly_charges_bin'] == 'High') * 0.25 +
            (df['tenure'] < 12) * 0.2
        ) / 1.2
        
        df['churn'] = np.random.binomial(1, churn_proba)
        
        return df
    
    def explore_data(self, df):
        """Comprehensive data exploration"""
        print("=== Data Exploration ===")
        print(f"Dataset shape: {df.shape}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        print(f"\nData Types:")
        print(df.dtypes)
        
        print(f"\nMissing Values:")
        print(df.isnull().sum())
        
        print(f"\nChurn Distribution:")
        print(df['churn'].value_counts())
        print(f"Churn Rate: {df['churn'].mean():.2%}")
        
        # Visualization
        plt.figure(figsize=(15, 10))
        
        # Churn distribution
        plt.subplot(2, 3, 1)
        df['churn'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Churn Distribution')
        plt.xlabel('Churn')
        plt.ylabel('Count')
        
        # Numerical features distribution
        plt.subplot(2, 3, 2)
        sns.histplot(data=df, x='monthly_charges', hue='churn', kde=True)
        plt.title('Monthly Charges by Churn')
        
        plt.subplot(2, 3, 3)
        sns.boxplot(data=df, x='churn', y='tenure')
        plt.title('Tenure by Churn')
        
        # Categorical features
        plt.subplot(2, 3, 4)
        pd.crosstab(df['contract'], df['churn'], normalize='index').plot(kind='bar', stacked=True)
        plt.title('Churn Rate by Contract Type')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 5)
        pd.crosstab(df['internet_service'], df['churn'], normalize='index').plot(kind='bar', stacked=True)
        plt.title('Churn Rate by Internet Service')
        plt.xticks(rotation=45)
        
        # Correlation heatmap
        plt.subplot(2, 3, 6)
        numerical_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlations')
        
        plt.tight_layout()
        plt.savefig('churn_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def preprocess_data(self, df):
        """Data preprocessing and feature engineering"""
        print("\n=== Data Preprocessing ===")
        
        # Create copy and store feature names
        df_processed = df.copy()
        self.feature_names = df_processed.columns.tolist()
        
        # Create new features
        df_processed['tenure_group'] = pd.cut(
            df_processed['tenure'], 
            bins=[0, 12, 24, 48, 72], 
            labels=['New', 'Regular', 'Loyal', 'VIP']
        )
        
        df_processed['charge_tenure_ratio'] = df_processed['monthly_charges'] / df_processed['tenure'].replace(0, 1)
        
        # Encode categorical variables
        categorical_columns = ['contract', 'internet_service', 'online_security', 
                              'tech_support', 'monthly_charges_bin', 'tenure_group']
        
        for col in categorical_columns:
            if col in df_processed.columns:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(
                    df_processed[col].astype(str)
                )
        
        # Separate features and target
        X = df_processed.drop('churn', axis=1)
        y = df_processed['churn']
        
        print(f"Processed features: {X.shape[1]}")
        print(f"Feature names: {list(X.columns)}")
        
        return X, y
    
    def train_model(self, X, y):
        """Model training with hyperparameter tuning"""
        print("\n=== Model Training ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Hyperparameter tuning
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        }
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.scaler = self.model.named_steps['scaler']
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.named_steps['classifier'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Feature Importance:")
        print(feature_importance.head(10))
        
        return X_test, y_test, accuracy
    
    def visualize_results(self, X_test, y_test):
        """Visualize model results"""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        plt.figure(figsize=(15, 5))
        
        # Confusion Matrix
        plt.subplot(1, 3, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Feature Importance
        plt.subplot(1, 3, 2)
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': self.model.named_steps['classifier'].feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Top 10 Feature Importance')
        plt.xlabel('Importance')
        
        # Prediction Distribution
        plt.subplot(1, 3, 3)
        plt.hist(y_pred_proba[y_test == 0], alpha=0.7, label='Not Churned', bins=20)
        plt.hist(y_pred_proba[y_test == 1], alpha=0.7, label='Churned', bins=20)
        plt.title('Prediction Probability Distribution')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filename='churn_predictor.pkl'):
        """Save trained model and preprocessing objects"""
        model_artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_artifacts, filename)
        print(f"\nModel saved as {filename}")
    
    def load_model(self, filename='churn_predictor.pkl'):
        """Load trained model and preprocessing objects"""
        model_artifacts = joblib.load(filename)
        
        self.model = model_artifacts['model']
        self.scaler = model_artifacts['scaler']
        self.label_encoders = model_artifacts['label_encoders']
        self.feature_names = model_artifacts['feature_names']
        
        print(f"Model loaded from {filename}")
    
    def predict_new_customer(self, customer_data):
        """Predict churn probability for new customer"""
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load model first.")
        
        # Convert to DataFrame and preprocess
        customer_df = pd.DataFrame([customer_data])
        
        # Apply same preprocessing
        for col, encoder in self.label_encoders.items():
            if col in customer_df.columns:
                # Handle unseen labels
                customer_df[col] = customer_df[col].apply(
                    lambda x: x if x in encoder.classes_ else encoder.classes_[0]
                )
                customer_df[col] = encoder.transform(customer_df[col])
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in customer_df.columns and feature != 'churn':
                customer_df[feature] = 0
        
        # Reorder columns to match training
        customer_df = customer_df[self.feature_names if 'churn' not in self.feature_names 
                                 else [f for f in self.feature_names if f != 'churn']]
        
        # Predict
        churn_probability = self.model.predict_proba(customer_df)[0, 1]
        prediction = self.model.predict(customer_df)[0]
        
        return {
            'churn_probability': churn_probability,
            'prediction': 'Churn' if prediction == 1 else 'No Churn',
            'confidence': max(churn_probability, 1 - churn_probability)
        }

def run_end_to_end_project():
    """Execute the complete end-to-end project"""
    print("=== Customer Churn Prediction - End-to-End Project ===\n")
    
    # Initialize predictor
    predictor = CustomerChurnPredictor()
    
    # 1. Generate and explore data
    print("Step 1: Data Generation and Exploration")
    df = predictor.generate_synthetic_data(10000)
    predictor.explore_data(df)
    
    # 2. Preprocess data
    print("Step 2: Data Preprocessing")
    X, y = predictor.preprocess_data(df)
    
    # 3. Train model
    print("Step 3: Model Training")
    X_test, y_test, accuracy = predictor.train_model(X, y)
    
    # 4. Visualize results
    print("Step 4: Results Visualization")
    predictor.visualize_results(X_test, y_test)
    
    # 5. Save model
    print("Step 5: Model Deployment")
    predictor.save_model()
    
    # 6. Demonstrate prediction on new data
    print("Step 6: Prediction Demonstration")
    new_customer = {
        'age': 45,
        'monthly_charges': 85.0,
        'total_charges': 2550.0,
        'tenure': 15,
        'contract': 'Month-to-month',
        'internet_service': 'Fiber optic',
        'online_security': 'No',
        'tech_support': 'No',
        'monthly_charges_bin': 'High'
    }
    
    prediction = predictor.predict_new_customer(new_customer)
    print(f"\nNew Customer Prediction:")
    print(f"Churn Probability: {prediction['churn_probability']:.2%}")
    print(f"Prediction: {prediction['prediction']}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    
    print("\n=== Project Completed Successfully ===")
    return predictor

# Execute the end-to-end project
if __name__ == "__main__":
    churn_predictor = run_end_to_end_project()