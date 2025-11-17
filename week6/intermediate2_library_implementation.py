import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

class AdvancedDataOperations:
    """Implement SQL-like joins and aggregations using Python libraries"""
    
    def __init__(self):
        self.dfs = {}
    
    def create_sample_dataframes(self):
        """Create sample DataFrames for demonstration"""
        # Employees DataFrame
        self.dfs['employees'] = pd.DataFrame({
            'emp_id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'dept_id': [1, 1, 2, 2, 3],
            'salary': [50000, 60000, 55000, 70000, 80000],
            'manager_id': [None, 1, 1, 3, None]
        })
        
        # Departments DataFrame
        self.dfs['departments'] = pd.DataFrame({
            'dept_id': [1, 2, 3],
            'dept_name': ['Engineering', 'Marketing', 'Sales'],
            'budget': [1000000, 500000, 300000]
        })
        
        # Sales DataFrame
        self.dfs['sales'] = pd.DataFrame({
            'sale_id': range(1, 21),
            'emp_id': [1, 2, 3, 4, 5] * 4,
            'amount': np.random.randint(1000, 5000, 20),
            'quarter': ['Q1', 'Q1', 'Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q2', 'Q2', 'Q2',
                       'Q3', 'Q3', 'Q3', 'Q3', 'Q3', 'Q4', 'Q4', 'Q4', 'Q4', 'Q4']
        })
        
        return self.dfs
    
    def pandas_advanced_joins(self):
        """Implement advanced joins using pandas"""
        employees = self.dfs['employees']
        departments = self.dfs['departments']
        sales = self.dfs['sales']
        
        # Multiple table join (equivalent to SQL INNER JOIN)
        joined_data = employees.merge(
            departments, 
            left_on='dept_id', 
            right_on='dept_id'
        ).merge(
            sales,
            left_on='emp_id',
            right_on='emp_id'
        )
        
        # Self join (employees and their managers)
        manager_join = employees.merge(
            employees[['emp_id', 'name', 'salary']],
            left_on='manager_id',
            right_on='emp_id',
            suffixes=('', '_manager')
        )
        
        return joined_data, manager_join
    
    def pandas_advanced_aggregations(self):
        """Implement advanced aggregations using pandas"""
        joined_data, _ = self.pandas_advanced_joins()
        
        # Complex aggregations with multiple groups
        dept_aggregations = joined_data.groupby(['dept_name', 'quarter']).agg({
            'amount': ['sum', 'mean', 'count', 'std'],
            'salary': 'mean',
            'emp_id': 'nunique'
        }).round(2)
        
        # Window functions equivalent
        joined_data['dept_salary_rank'] = joined_data.groupby('dept_name')['salary'].rank(ascending=False)
        joined_data['rolling_avg_sales'] = joined_data.groupby('emp_id')['amount'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        # Pivot table (similar to GROUPING SETS)
        pivot_table = pd.pivot_table(
            joined_data,
            values='amount',
            index='dept_name',
            columns='quarter',
            aggfunc=['sum', 'mean'],
            margins=True
        )
        
        return dept_aggregations, joined_data, pivot_table
    
    def sklearn_analytics(self):
        """Use scikit-learn for advanced analytics on joined data"""
        joined_data, _ = self.pandas_advanced_joins()
        
        # Prepare features for clustering
        features = joined_data.groupby('emp_id').agg({
            'amount': ['sum', 'mean', 'count'],
            'salary': 'mean'
        }).fillna(0)
        
        features.columns = ['total_sales', 'avg_sale', 'sale_count', 'salary']
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        features['cluster'] = clusters
        
        return features, kmeans
    
    def pytorch_analytics(self):
        """Use PyTorch for neural network analysis on aggregated data"""
        features, _ = self.sklearn_analytics()
        
        # Prepare data for neural network
        X = torch.FloatTensor(features[['total_sales', 'avg_sale', 'sale_count', 'salary']].values)
        y = torch.FloatTensor(features['cluster'].values)
        
        # Simple neural network for demonstration
        class SalesClusterNet(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(SalesClusterNet, self).__init__()
                self.layer1 = nn.Linear(input_size, hidden_size)
                self.layer2 = nn.Linear(hidden_size, hidden_size)
                self.output = nn.Linear(hidden_size, output_size)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.relu(self.layer1(x))
                x = self.relu(self.layer2(x))
                x = self.output(x)
                return x
        
        # Initialize and run model
        model = SalesClusterNet(input_size=4, hidden_size=8, output_size=3)
        
        with torch.no_grad():
            predictions = model(X)
            predicted_clusters = torch.argmax(predictions, dim=1)
        
        features['nn_cluster'] = predicted_clusters.numpy()
        
        return features, model
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("=== ADVANCED DATA OPERATIONS USING PYTHON LIBRARIES ===\n")
        
        # Create sample data
        self.create_sample_dataframes()
        
        print("1. PANDAS ADVANCED JOINS:")
        joined_data, manager_join = self.pandas_advanced_joins()
        print(f"   - Joined dataset shape: {joined_data.shape}")
        print(f"   - Manager self-join sample:")
        print(manager_join[['name', 'salary', 'name_manager', 'salary_manager']].head())
        
        print("\n2. PANDAS ADVANCED AGGREGATIONS:")
        dept_agg, window_data, pivot = self.pandas_advanced_aggregations()
        print("   - Department aggregations:")
        print(dept_agg.head())
        print(f"   - Window functions applied: {list(window_data.columns)}")
        
        print("\n3. SCIKIT-LEARN CLUSTERING:")
        features, kmeans = self.sklearn_analytics()
        print(f"   - Clusters identified: {len(np.unique(kmeans.labels_))}")
        print(f"   - Cluster distribution:")
        print(features['cluster'].value_counts().sort_index())
        
        print("\n4. PYTORCH NEURAL NETWORK:")
        nn_features, model = self.pytorch_analytics()
        print(f"   - Neural network architecture: {model}")
        print(f"   - NN cluster distribution:")
        print(nn_features['nn_cluster'].value_counts().sort_index())
        
        # Compare clustering results
        agreement = (features['cluster'] == nn_features['nn_cluster']).mean()
        print(f"\n5. CLUSTERING AGREEMENT: {agreement:.2%}")
        
        return {
            'joined_data': joined_data,
            'features': features,
            'nn_features': nn_features,
            'model': model
        }

# Execute the analysis
if __name__ == "__main__":
    analyzer = AdvancedDataOperations()
    results = analyzer.generate_report()