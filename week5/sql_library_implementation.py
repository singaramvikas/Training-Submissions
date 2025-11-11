"""
Intermediate 2: Implement SQL Basics using appropriate library (Pandas).

We'll use Pandas, which provides SQL-like operations for data manipulation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union

class PandasSQLImplementation:
    """
    A class that implements SQL basics using Pandas library.
    This demonstrates how Pandas can be used for SQL-like operations.
    """
    
    def __init__(self):
        self.dataframes = {}
    
    def create_table(self, table_name: str, data: Union[List[Dict], pd.DataFrame]):
        """CREATE TABLE equivalent"""
        if isinstance(data, list):
            self.dataframes[table_name] = pd.DataFrame(data)
        else:
            self.dataframes[table_name] = data
        print(f"Created table '{table_name}' with {len(self.dataframes[table_name])} rows")
    
    def select(self, table_name: str, columns: List[str] = None, 
               where: Dict = None, order_by: str = None, 
               ascending: bool = True, limit: int = None) -> pd.DataFrame:
        """SELECT query implementation"""
        if table_name not in self.dataframes:
            raise ValueError(f"Table '{table_name}' not found")
        
        df = self.dataframes[table_name].copy()
        
        # SELECT specific columns
        if columns and columns != ['*']:
            df = df[columns]
        
        # WHERE clause
        if where:
            for col, condition in where.items():
                if isinstance(condition, tuple) and len(condition) == 2:
                    # Range condition
                    df = df[(df[col] >= condition[0]) & (df[col] <= condition[1])]
                elif callable(condition):
                    # Function condition
                    df = df[df.apply(condition, axis=1)]
                else:
                    # Equality condition
                    df = df[df[col] == condition]
        
        # ORDER BY clause
        if order_by and order_by in df.columns:
            df = df.sort_values(by=order_by, ascending=ascending)
        
        # LIMIT clause
        if limit:
            df = df.head(limit)
        
        return df
    
    def insert(self, table_name: str, data: Dict):
        """INSERT INTO implementation"""
        if table_name not in self.dataframes:
            raise ValueError(f"Table '{table_name}' not found")
        
        new_row = pd.DataFrame([data])
        self.dataframes[table_name] = pd.concat([self.dataframes[table_name], new_row], ignore_index=True)
        print(f"Inserted 1 row into '{table_name}'")
    
    def update(self, table_name: str, updates: Dict, where: Dict):
        """UPDATE implementation"""
        if table_name not in self.dataframes:
            raise ValueError(f"Table '{table_name}' not found")
        
        mask = pd.Series([True] * len(self.dataframes[table_name]))
        for col, condition in where.items():
            if isinstance(condition, tuple) and len(condition) == 2:
                mask &= (self.dataframes[table_name][col] >= condition[0]) & (self.dataframes[table_name][col] <= condition[1])
            else:
                mask &= (self.dataframes[table_name][col] == condition)
        
        for col, new_value in updates.items():
            self.dataframes[table_name].loc[mask, col] = new_value
        
        print(f"Updated {mask.sum()} rows in '{table_name}'")
    
    def delete(self, table_name: str, where: Dict):
        """DELETE implementation"""
        if table_name not in self.dataframes:
            raise ValueError(f"Table '{table_name}' not found")
        
        mask = pd.Series([True] * len(self.dataframes[table_name]))
        for col, condition in where.items():
            if isinstance(condition, tuple) and len(condition) == 2:
                mask &= (self.dataframes[table_name][col] >= condition[0]) & (self.dataframes[table_name][col] <= condition[1])
            else:
                mask &= (self.dataframes[table_name][col] == condition)
        
        self.dataframes[table_name] = self.dataframes[table_name][~mask]
        print(f"Deleted {mask.sum()} rows from '{table_name}'")
    
    def group_by(self, table_name: str, group_col: str, agg_operations: Dict):
        """GROUP BY implementation"""
        if table_name not in self.dataframes:
            raise ValueError(f"Table '{table_name}' not found")
        
        return self.dataframes[table_name].groupby(group_col).agg(agg_operations)
    
    def inner_join(self, left_table: str, right_table: str, on: str):
        """INNER JOIN implementation"""
        if left_table not in self.dataframes or right_table not in self.dataframes:
            raise ValueError("One or both tables not found")
        
        return pd.merge(self.dataframes[left_table], self.dataframes[right_table], on=on)

def demonstrate_pandas_sql():
    sql_engine = PandasSQLImplementation()
    
    # Create sample tables
    employees_data = [
        {'id': 1, 'name': 'John Doe', 'department': 'Engineering', 'salary': 75000},
        {'id': 2, 'name': 'Jane Smith', 'department': 'Marketing', 'salary': 65000},
        {'id': 3, 'name': 'Bob Johnson', 'department': 'Engineering', 'salary': 80000},
        {'id': 4, 'name': 'Alice Brown', 'department': 'HR', 'salary': 55000},
        {'id': 5, 'name': 'Charlie Wilson', 'department': 'Marketing', 'salary': 70000}
    ]
    
    departments_data = [
        {'dept_id': 1, 'department': 'Engineering', 'manager': 'Tech Director'},
        {'dept_id': 2, 'department': 'Marketing', 'manager': 'Marketing Head'},
        {'dept_id': 3, 'department': 'HR', 'manager': 'HR Director'}
    ]
    
    sql_engine.create_table('employees', employees_data)
    sql_engine.create_table('departments', departments_data)
    
    print("PANDAS SQL IMPLEMENTATION DEMONSTRATION")
    print("=" * 50)
    
    # SELECT all employees
    print("\n1. SELECT * FROM employees:")
    result = sql_engine.select('employees')
    print(result.to_string(index=False))
    
    # SELECT with WHERE clause
    print("\n2. SELECT name, salary FROM employees WHERE department = 'Engineering':")
    result = sql_engine.select(
        'employees', 
        columns=['name', 'salary'],
        where={'department': 'Engineering'}
    )
    print(result.to_string(index=False))
    
    # SELECT with ORDER BY
    print("\n3. SELECT * FROM employees ORDER BY salary DESC:")
    result = sql_engine.select(
        'employees',
        order_by='salary',
        ascending=False
    )
    print(result.to_string(index=False))
    
    # GROUP BY with aggregation
    print("\n4. SELECT department, AVG(salary) FROM employees GROUP BY department:")
    result = sql_engine.group_by(
        'employees',
        'department',
        {'salary': 'mean'}
    )
    print(result)
    
    # INNER JOIN
    print("\n5. SELECT * FROM employees INNER JOIN departments ON employees.department = departments.department:")
    result = sql_engine.inner_join('employees', 'departments', 'department')
    print(result.to_string(index=False))
    
    # INSERT new employee
    print("\n6. INSERT INTO employees VALUES (6, 'David Lee', 'Engineering', 72000):")
    sql_engine.insert('employees', {
        'id': 6, 
        'name': 'David Lee', 
        'department': 'Engineering', 
        'salary': 72000
    })
    
    # UPDATE employee salary
    print("\n7. UPDATE employees SET salary = 85000 WHERE name = 'Bob Johnson':")
    sql_engine.update(
        'employees',
        updates={'salary': 85000},
        where={'name': 'Bob Johnson'}
    )
    
    # Show final state
    print("\n8. Final employee table:")
    result = sql_engine.select('employees')
    print(result.to_string(index=False))

if __name__ == "__main__":
    demonstrate_pandas_sql()