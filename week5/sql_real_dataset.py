"""
Intermediate 1: Apply SQL Basics on a real dataset and explain results.

We'll use the Titanic dataset to demonstrate SQL-like operations on real data.
"""

import pandas as pd
import numpy as np
from collections import defaultdict

class SQLOnRealData:
    def __init__(self):
        # Load Titanic dataset
        try:
            self.df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
        except:
            # Fallback: create sample data if download fails
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample Titanic-like data if download fails"""
        data = {
            'PassengerId': range(1, 892),
            'Survived': np.random.randint(0, 2, 891),
            'Pclass': np.random.randint(1, 4, 891),
            'Name': [f'Passenger {i}' for i in range(1, 892)],
            'Sex': np.random.choice(['male', 'female'], 891),
            'Age': np.random.normal(30, 15, 891).astype(int),
            'SibSp': np.random.randint(0, 5, 891),
            'Parch': np.random.randint(0, 4, 891),
            'Fare': np.random.exponential(50, 891),
            'Embarked': np.random.choice(['C', 'Q', 'S'], 891)
        }
        self.df = pd.DataFrame(data)
        print("Using generated sample data (real dataset unavailable)")
    
    def execute_query(self, query_type, **kwargs):
        """Execute different types of SQL-like queries"""
        if query_type == "select_where":
            return self._select_where(**kwargs)
        elif query_type == "group_by":
            return self._group_by(**kwargs)
        elif query_type == "aggregate":
            return self._aggregate(**kwargs)
        elif query_type == "join_simulation":
            return self._join_simulation()
    
    def _select_where(self, conditions):
        """Simulate SELECT with WHERE clause"""
        result = self.df.copy()
        for col, value in conditions.items():
            if col in result.columns:
                if isinstance(value, tuple) and len(value) == 2:
                    # Range condition (min, max)
                    result = result[(result[col] >= value[0]) & (result[col] <= value[1])]
                else:
                    # Equality condition
                    result = result[result[col] == value]
        return result
    
    def _group_by(self, group_col, agg_col, agg_func):
        """Simulate GROUP BY with aggregation"""
        return self.df.groupby(group_col)[agg_col].agg(agg_func).to_dict()
    
    def _aggregate(self, operations):
        """Perform multiple aggregations"""
        results = {}
        for op_name, (col, func) in operations.items():
            if col in self.df.columns:
                results[op_name] = func(self.df[col])
        return results
    
    def _join_simulation(self):
        """Simulate a JOIN operation by creating related tables"""
        # Create a "tickets" table related to passengers
        tickets_data = {
            'PassengerId': self.df['PassengerId'].sample(500).values,
            'TicketNumber': [f'TKT{1000+i}' for i in range(500)],
            'TicketClass': np.random.choice(['First', 'Second', 'Third'], 500)
        }
        tickets_df = pd.DataFrame(tickets_data)
        
        # Simulate INNER JOIN
        joined = pd.merge(self.df, tickets_df, on='PassengerId', how='inner')
        return joined

def analyze_real_dataset():
    analyzer = SQLOnRealData()
    
    print("REAL DATASET ANALYSIS - TITANIC DATA")
    print("=" * 50)
    print(f"Dataset shape: {analyzer.df.shape}")
    print(f"Columns: {list(analyzer.df.columns)}")
    
    # Query 1: Filter data (WHERE clause)
    print("\n1. SELECT * FROM titanic WHERE Pclass = 1 AND Sex = 'female':")
    first_class_women = analyzer.execute_query(
        "select_where", 
        conditions={'Pclass': 1, 'Sex': 'female'}
    )
    print(f"   Found {len(first_class_women)} first class female passengers")
    print(f"   Survival rate: {first_class_women['Survived'].mean():.2%}")
    
    # Query 2: Group by with aggregation
    print("\n2. SELECT Pclass, AVG(Fare) FROM titanic GROUP BY Pclass:")
    avg_fare_by_class = analyzer.execute_query(
        "group_by",
        group_col='Pclass',
        agg_col='Fare',
        agg_func='mean'
    )
    for pclass, avg_fare in avg_fare_by_class.items():
        print(f"   Class {pclass}: ${avg_fare:.2f} average fare")
    
    # Query 3: Multiple aggregations
    print("\n3. Aggregate statistics:")
    stats = analyzer.execute_query("aggregate", operations={
        'avg_age': ('Age', lambda x: x.mean()),
        'survival_rate': ('Survived', lambda x: x.mean()),
        'max_fare': ('Fare', lambda x: x.max())
    })
    for stat_name, value in stats.items():
        print(f"   {stat_name}: {value:.2f}")
    
    # Query 4: Age range filter
    print("\n4. SELECT * FROM titanic WHERE Age BETWEEN 20 AND 30:")
    young_adults = analyzer.execute_query(
        "select_where",
        conditions={'Age': (20, 30)}
    )
    print(f"   Found {len(young_adults)} passengers aged 20-30")
    print(f"   Their survival rate: {young_adults['Survived'].mean():.2%}")

if __name__ == "__main__":
    analyze_real_dataset()