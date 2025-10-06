import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

def analyze_iris_dataset():
    """Apply Python basics to analyze the Iris dataset"""
    
    # Load dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("=== Iris Dataset Analysis ===")
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nDataset Info:")
    print(df.info())
    
    print(f"\nBasic Statistics:")
    print(df.describe())
    
    print(f"\nSpecies Distribution:")
    print(df['species_name'].value_counts())
    
    # Data visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='sepal length (cm)', hue='species_name', kde=True)
    plt.title('Sepal Length Distribution')
    
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='species_name')
    plt.title('Sepal Length vs Width')
    
    plt.subplot(2, 2, 3)
    sns.boxplot(data=df, x='species_name', y='petal length (cm)')
    plt.title('Petal Length by Species')
    
    plt.subplot(2, 2, 4)
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('iris_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Basic insights
    print("\n=== Key Insights ===")
    print("1. Setosa species has distinctly smaller petals")
    print("2. Petal measurements show stronger correlation with species")
    print("3. Virginica has the largest sepal and petal measurements")
    print("4. Features show good separation between species")

# Run analysis
analyze_iris_dataset()