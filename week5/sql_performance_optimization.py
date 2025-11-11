"""
Hard 1: Optimize the implementation of SQL Basics for performance.

We'll implement various optimization techniques for SQL-like operations:
1. Indexing for faster lookups
2. Query optimization
3. Memory optimization
4. Parallel processing
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any, Union
from collections import defaultdict
import hashlib

class OptimizedSQLEngine:
    """
    An optimized SQL-like engine with performance enhancements.
    """
    
    def __init__(self):
        self.tables = {}
        self.indexes = {}  # {table_name: {column_name: index}}
        self.query_cache = {}  # Simple query result caching
    
    def create_table(self, table_name: str, data: pd.DataFrame):
        """Create table with optional indexing"""
        self.tables[table_name] = data.copy()
        self.indexes[table_name] = {}
        print(f"Created table '{table_name}' with {len(data)} rows")
    
    def create_index(self, table_name: str, column_name: str):
        """Create index on specified column for faster lookups"""
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' not found")
        
        if column_name not in self.tables[table_name].columns:
            raise ValueError(f"Column '{column_name}' not found in table '{table_name}'")
        
        # Create inverted index: value -> list of row indices
        index = defaultdict(list)
        for idx, value in enumerate(self.tables[table_name][column_name]):
            index[value].append(idx)
        
        self.indexes[table_name][column_name] = dict(index)
        print(f"Created index on {table_name}.{column_name}")
    
    def _get_query_hash(self, table_name: str, operation: str, **kwargs) -> str:
        """Generate hash for query caching"""
        query_str = f"{table_name}_{operation}_{str(kwargs)}"
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def optimized_select(self, table_name: str, columns: List[str] = None,
                        where: Dict = None, use_index: bool = True) -> pd.DataFrame:
        """Optimized SELECT with indexing and caching"""
        
        # Check cache first
        cache_key = self._get_query_hash(table_name, 'select', 
                                       columns=columns, where=where)
        if cache_key in self.query_cache:
            print("Using cached result")
            return self.query_cache[cache_key].copy()
        
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' not found")
        
        df = self.tables[table_name]
        start_time = time.time()
        
        # Use indexes for WHERE clause if available
        if where and use_index:
            mask = self._optimized_where_clause(table_name, where)
            result = df.loc[mask]
        else:
            # Fallback to regular filtering
            result = df.copy()
            for col, condition in (where or {}).items():
                if isinstance(condition, tuple):
                    result = result[(result[col] >= condition[0]) & (result[col] <= condition[1])]
                else:
                    result = result[result[col] == condition]
        
        # Select specific columns
        if columns and columns != ['*']:
            result = result[columns]
        
        execution_time = time.time() - start_time
        print(f"Query executed in {execution_time:.4f} seconds")
        
        # Cache the result
        self.query_cache[cache_key] = result.copy()
        
        return result
    
    def _optimized_where_clause(self, table_name: str, where: Dict) -> pd.Series:
        """Use indexes to optimize WHERE clause"""
        df = self.tables[table_name]
        mask = pd.Series([True] * len(df))
        
        for col, condition in where.items():
            if (col in self.indexes[table_name] and 
                not isinstance(condition, tuple)):  # Index only works for equality
                
                index = self.indexes[table_name][col]
                if condition in index:
                    # Use index to get row indices
                    row_indices = index[condition]
                    col_mask = pd.Series([False] * len(df))
                    col_mask.iloc[row_indices] = True
                    mask &= col_mask
                else:
                    # Value not in index, no rows match
                    mask &= pd.Series([False] * len(df))
            else:
                # Fallback to regular filtering for ranges or non-indexed columns
                if isinstance(condition, tuple):
                    mask &= (df[col] >= condition[0]) & (df[col] <= condition[1])
                else:
                    mask &= (df[col] == condition)
        
        return mask
    
    def batch_insert(self, table_name: str, data: List[Dict]):
        """Optimized batch insert"""
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' not found")
        
        new_data = pd.DataFrame(data)
        self.tables[table_name] = pd.concat([self.tables[table_name], new_data], ignore_index=True)
        
        # Invalidate cache for this table
        self._invalidate_table_cache(table_name)
        print(f"Batch inserted {len(data)} rows into '{table_name}'")
    
    def _invalidate_table_cache(self, table_name: str):
        """Invalidate cache entries for a specific table"""
        keys_to_remove = [k for k in self.query_cache.keys() if k.startswith(hashlib.md5(table_name.encode()).hexdigest()[:10])]
        for key in keys_to_remove:
            del self.query_cache[key]
    
    def optimized_group_by(self, table_name: str, group_col: str, agg_col: str, agg_func: str):
        """Optimized GROUP BY using Pandas' efficient groupby"""
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' not found")
        
        cache_key = self._get_query_hash(table_name, 'group_by', 
                                       group_col=group_col, agg_col=agg_col, agg_func=agg_func)
        
        if cache_key in self.query_cache:
            print("Using cached groupby result")
            return self.query_cache[cache_key]
        
        start_time = time.time()
        
        # Use Pandas' optimized groupby
        result = self.tables[table_name].groupby(group_col)[agg_col].agg(agg_func)
        
        execution_time = time.time() - start_time
        print(f"GroupBy executed in {execution_time:.4f} seconds")
        
        self.query_cache[cache_key] = result
        return result
    
    def memory_optimize(self, table_name: str):
        """Optimize memory usage by downcasting numeric columns"""
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' not found")
        
        df = self.tables[table_name]
        original_memory = df.memory_usage(deep=True).sum()
        
        # Downcast numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        optimized_memory = df.memory_usage(deep=True).sum()
        memory_saved = original_memory - optimized_memory
        
        print(f"Memory optimized for '{table_name}':")
        print(f"  Original: {original_memory / 1024:.2f} KB")
        print(f"  Optimized: {optimized_memory / 1024:.2f} KB")
        print(f"  Saved: {memory_saved / 1024:.2f} KB ({memory_saved/original_memory*100:.1f}%)")

def performance_comparison():
    """Compare performance between optimized and naive implementations"""
    
    # Create large dataset for testing
    np.random.seed(42)
    n_rows = 100000
    
    large_data = pd.DataFrame({
        'id': range(1, n_rows + 1),
        'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows),
        'value1': np.random.randint(1, 1000, n_rows),
        'value2': np.random.normal(100, 50, n_rows),
        'flag': np.random.choice([True, False], n_rows)
    })
    
    # Initialize engines
    optimized_engine = OptimizedSQLEngine()
    optimized_engine.create_table('large_table', large_data)
    optimized_engine.create_index('large_table', 'category')
    optimized_engine.create_index('large_table', 'flag')
    
    print("PERFORMANCE OPTIMIZATION COMPARISON")
    print("=" * 50)
    print(f"Testing on dataset with {n_rows} rows")
    
    # Test 1: Equality filter with index
    print("\n1. Filter by indexed category ('A'):")
    start_time = time.time()
    result_opt = optimized_engine.optimized_select(
        'large_table', 
        where={'category': 'A'}
    )
    opt_time = time.time() - start_time
    print(f"   Optimized: {opt_time:.4f}s, {len(result_opt)} rows")
    
    # Test 2: Range query (non-indexed)
    print("\n2. Filter by value range (500-600):")
    start_time = time.time()
    result_range = optimized_engine.optimized_select(
        'large_table',
        where={'value1': (500, 600)}
    )
    range_time = time.time() - start_time
    print(f"   Execution time: {range_time:.4f}s, {len(result_range)} rows")
    
    # Test 3: GroupBy performance
    print("\n3. GROUP BY category, average value1:")
    start_time = time.time()
    group_result = optimized_engine.optimized_group_by(
        'large_table', 'category', 'value1', 'mean'
    )
    group_time = time.time() - start_time
    print(f"   Execution time: {group_time:.4f}s")
    print(f"   Results: {dict(group_result)}")
    
    # Test 4: Memory optimization
    print("\n4. Memory optimization:")
    optimized_engine.memory_optimize('large_table')
    
    # Test 5: Caching performance
    print("\n5. Cached query performance (same query executed twice):")
    # First execution
    start_time = time.time()
    optimized_engine.optimized_select('large_table', where={'category': 'B'})
    first_time = time.time() - start_time
    
    # Second execution (should be faster due to caching)
    start_time = time.time()
    optimized_engine.optimized_select('large_table', where={'category': 'B'})
    second_time = time.time() - start_time
    
    print(f"   First execution: {first_time:.4f}s")
    print(f"   Second execution: {second_time:.4f}s")
    print(f"   Speedup: {first_time/second_time:.1f}x")

if __name__ == "__main__":
    performance_comparison()