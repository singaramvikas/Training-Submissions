"""
Intermediate 1: Apply Pandas Fundamentals on a real dataset and explain results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_titanic_dataset():
    """
    Analyze the Titanic dataset to demonstrate real-world Pandas applications
    """
    print("REAL DATASET ANALYSIS: Titanic Survival")
    print("=" * 50)
    
    # Load the dataset
    try:
        # Try to load from online source
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)
    except:
        # Fallback: create sample data
        print("Downloading dataset failed. Using sample data...")
        df = create_sample_titanic_data()
    
    print("1. Dataset Overview:")
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nBasic info:")
    print(df.info())
    
    print("\n2. Data Quality Check:")
    print("Missing values per column:")
    missing_data = df.isnull().sum()
    print(missing_data[missing_data > 0])
    
    print("\n3. Survival Analysis:")
    survival_rate = df['Survived'].value_counts(normalize=True) * 100
    print(f"Survival Rate: {survival_rate[1]:.2f}% survived")
    
    print("\n4. Gender-based Survival:")
    gender_survival = df.groupby('Sex')['Survived'].mean() * 100
    print(gender_survival.round(2))
    
    print("\n5. Class-based Analysis:")
    class_survival = df.groupby('Pclass')['Survived'].agg(['mean', 'count'])
    class_survival['mean'] = class_survival['mean'] * 100
    print(class_survival.round(2))
    
    print("\n6. Age Analysis:")
    print(f"Average age: {df['Age'].mean():.2f}")
    print(f"Age range: {df['Age'].min():.2f} - {df['Age'].max():.2f}")
    
    # Create age groups
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 100], 
                           labels=['Child', 'Young Adult', 'Adult', 'Senior'])
    
    age_survival = df.groupby('AgeGroup')['Survived'].mean() * 100
    print("\n7. Survival by Age Group:")
    print(age_survival.round(2))
    
    print("\n8. Family Size Analysis:")
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    family_survival = df.groupby('FamilySize')['Survived'].mean() * 100
    print(family_survival.round(2))
    
    # Generate insights
    print("\n" + "="*50)
    print("KEY INSIGHTS:")
    print("="*50)
    print("1. Overall survival rate was low (around 38%)")
    print("2. Females had significantly higher survival rates")
    print("3. Higher class passengers had better survival chances")
    print("4. Children had relatively good survival rates")
    print("5. Medium family sizes (2-4) had better survival rates")
    
    return df

def create_sample_titanic_data():
    """Create sample Titanic data if download fails"""
    np.random.seed(42)
    n_passengers = 200
    
    data = {
        'PassengerId': range(1, n_passengers + 1),
        'Survived': np.random.choice([0, 1], n_passengers, p=[0.62, 0.38]),
        'Pclass': np.random.choice([1, 2, 3], n_passengers, p=[0.25, 0.25, 0.5]),
        'Sex': np.random.choice(['male', 'female'], n_passengers, p=[0.65, 0.35]),
        'Age': np.random.normal(29, 14, n_passengers).clip(0.4, 80),
        'SibSp': np.random.poisson(0.5, n_passengers),
        'Parch': np.random.poisson(0.4, n_passengers),
        'Fare': np.random.exponential(32, n_passengers),
    }
    
    # Make some correlations realistic
    df = pd.DataFrame(data)
    df.loc[df['Pclass'] == 1, 'Fare'] *= 2
    df.loc[df['Pclass'] == 3, 'Fare'] *= 0.5
    df.loc[df['Sex'] == 'female', 'Survived'] = np.random.choice([0, 1], sum(df['Sex'] == 'female'), p=[0.25, 0.75])
    df.loc[df['Age'] < 18, 'Survived'] = np.random.choice([0, 1], sum(df['Age'] < 18), p=[0.4, 0.6])
    
    return df

if __name__ == "__main__":
    df = analyze_titanic_dataset()