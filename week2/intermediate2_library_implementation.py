"""
Intermediate 2: Implement Python Data Structures using appropriate library (Scikit-learn, PyTorch, etc.)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class LibraryDataStructures:
    """Demonstrate data structures using various Python libraries"""
    
    def __init__(self):
        self.results = {}
    
    def numpy_arrays_demo(self):
        """Demonstrate NumPy arrays as efficient data structures"""
        print("NUMPY ARRAYS DEMONSTRATION")
        print("=" * 30)
        
        # Create sample sales data
        sales_data = np.array([
            [100, 150, 200, 180],  # Product A quarterly sales
            [80, 120, 160, 140],   # Product B quarterly sales
            [200, 180, 220, 240]   # Product C quarterly sales
        ])
        
        print(f"Sales Data Shape: {sales_data.shape}")
        print(f"Sales Data:\n{sales_data}")
        
        # Array operations
        total_sales = np.sum(sales_data, axis=1)
        average_sales = np.mean(sales_data, axis=1)
        max_sales = np.max(sales_data, axis=1)
        
        print(f"\nTotal Sales per Product: {total_sales}")
        print(f"Average Sales per Product: {average_sales}")
        print(f"Max Quarterly Sales: {max_sales}")
        
        # Boolean indexing
        high_sales = sales_data[sales_data > 200]
        print(f"Sales above 200: {high_sales}")
        
        self.results['numpy'] = {
            'total_sales': total_sales.tolist(),
            'average_sales': average_sales.tolist(),
            'high_sales_count': len(high_sales)
        }
    
    def pandas_dataframes_demo(self):
        """Demonstrate Pandas DataFrames for structured data"""
        print("\nPANDAS DATAFRAMES DEMONSTRATION")
        print("=" * 35)
        
        # Create sample employee data
        data = {
            'EmployeeID': [101, 102, 103, 104, 105],
            'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'Department': ['HR', 'Engineering', 'Engineering', 'Marketing', 'HR'],
            'Salary': [50000, 80000, 75000, 60000, 55000],
            'Experience': [2, 5, 4, 3, 1]
        }
        
        df = pd.DataFrame(data)
        print("Employee DataFrame:")
        print(df)
        
        # DataFrame operations
        dept_stats = df.groupby('Department').agg({
            'Salary': ['mean', 'max', 'min'],
            'EmployeeID': 'count'
        }).round(2)
        
        print(f"\nDepartment Statistics:")
        print(dept_stats)
        
        # Filtering and sorting
        high_salary = df[df['Salary'] > 60000].sort_values('Salary', ascending=False)
        print(f"\nHigh Salary Employees:")
        print(high_salary)
        
        self.results['pandas'] = {
            'department_stats': dept_stats.to_dict(),
            'high_salary_count': len(high_salary)
        }
    
    def sklearn_demo(self):
        """Demonstrate scikit-learn for machine learning data structures"""
        print("\nSCIKIT-LEARN DEMONSTRATION")
        print("=" * 30)
        
        # Create sample classification data
        np.random.seed(42)
        X = np.random.randn(100, 4)  # 100 samples, 4 features
        y = np.random.randint(0, 2, 100)  # Binary classification
        
        # Feature names
        feature_names = ['Feature_A', 'Feature_B', 'Feature_C', 'Feature_D']
        
        # Create DataFrame for better visualization
        X_df = pd.DataFrame(X, columns=feature_names)
        X_df['Target'] = y
        
        print("Sample of the dataset:")
        print(X_df.head())
        
        # Preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Feature importance
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        
        print(f"\nFeature Importance:")
        for feature, importance in sorted(feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {importance:.4f}")
        
        self.results['sklearn'] = {
            'feature_importance': feature_importance,
            'train_score': model.score(X_train, y_train),
            'test_score': model.score(X_test, y_test)
        }
    
    def pytorch_demo(self):
        """Demonstrate PyTorch tensors and datasets"""
        print("\nPYTORCH DEMONSTRATION")
        print("=" * 25)
        
        # Create PyTorch tensors
        tensor_data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        tensor_labels = torch.tensor([0, 1, 0])
        
        print(f"Data Tensor: {tensor_data}")
        print(f"Shape: {tensor_data.shape}")
        print(f"Labels: {tensor_labels}")
        
        # Tensor operations
        tensor_sum = torch.sum(tensor_data, dim=1)
        tensor_mean = torch.mean(tensor_data, dim=0)
        
        print(f"\nSum along rows: {tensor_sum}")
        print(f"Mean along columns: {tensor_mean}")
        
        # Custom Dataset
        class CustomDataset(Dataset):
            def __init__(self, data, labels):
                self.data = data
                self.labels = labels
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]
        
        dataset = CustomDataset(tensor_data, tensor_labels)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        print(f"\nDataLoader batches:")
        for batch_idx, (data, labels) in enumerate(dataloader):
            print(f"  Batch {batch_idx}: data={data.shape}, labels={labels}")
        
        self.results['pytorch'] = {
            'tensor_sum': tensor_sum.tolist(),
            'tensor_mean': tensor_mean.tolist(),
            'dataset_size': len(dataset)
        }
    
    def run_all_demos(self):
        """Run all library demonstrations"""
        self.numpy_arrays_demo()
        self.pandas_dataframes_demo()
        self.sklearn_demo()
        self.pytorch_demo()
        
        print("\n" + "=" * 50)
        print("SUMMARY OF ALL LIBRARY DATA STRUCTURES")
        print("=" * 50)
        
        for library, result in self.results.items():
            print(f"\n{library.upper()}:")
            for key, value in result.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for subkey, subvalue in value.items():
                        print(f"    {subkey}: {subvalue}")
                else:
                    print(f"  {key}: {value}")

def main():
    """Main execution function"""
    demo = LibraryDataStructures()
    demo.run_all_demos()

if __name__ == "__main__":
    main()