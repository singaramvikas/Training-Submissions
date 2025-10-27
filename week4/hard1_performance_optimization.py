"""
Hard 1: Optimize the implementation of Pandas Fundamentals for performance
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class PandasOptimizer:
    """
    A class demonstrating various Pandas performance optimization techniques
    """
    
    def __init__(self, size: int = 1000000):
        self.size = size
        self.df = self._create_large_dataset()
    
    def _create_large_dataset(self) -> pd.DataFrame:
        """Create a large dataset for performance testing"""
        print(f"Creating dataset with {self.size:,} rows...")
        
        np.random.seed(42)
        data = {
            'user_id': np.arange(self.size),
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], self.size),
            'value1': np.random.normal(100, 50, self.size),
            'value2': np.random.exponential(10, self.size),
            'value3': np.random.randint(0, 1000, self.size),
            'timestamp': pd.date_range('2020-01-01', periods=self.size, freq='1min'),
            'flag': np.random.choice([True, False], self.size, p=[0.3, 0.7]),
        }
        
        # Add some missing values
        mask = np.random.random(self.size) < 0.1
        data['value1'] = np.where(mask, np.nan, data['value1'])
        
        return pd.DataFrame(data)
    
    def benchmark_operations(self):
        """Benchmark different optimization techniques"""
        print("\n" + "="*60)
        print("PANDAS PERFORMANCE OPTIMIZATION BENCHMARK")
        print("="*60)
        
        benchmarks = {}
        
        # 1. Basic operations vs Optimized
        benchmarks.update(self._benchmark_filtering())
        benchmarks.update(self._benchmark_groupby())
        benchmarks.update(self._benchmark_apply())
        benchmarks.update(self._benchmark_memory())
        benchmarks.update(self._benchmark_datatypes())
        
        # Print results
        print("\nBENCHMARK RESULTS:")
        print("="*40)
        for operation, results in benchmarks.items():
            print(f"\n{operation}:")
            for method, time_taken in results.items():
                print(f"  {method}: {time_taken:.4f}s")
    
    def _benchmark_filtering(self) -> Dict:
        """Benchmark different filtering methods"""
        print("\n1. FILTERING OPERATIONS:")
        
        results = {}
        
        # Method 1: Basic boolean indexing
        start = time.time()
        result1 = self.df[(self.df['value1'] > 100) & (self.df['value2'] < 20)]
        time1 = time.time() - start
        results["Basic boolean indexing"] = time1
        
        # Method 2: Using query (often faster for large datasets)
        start = time.time()
        result2 = self.df.query('value1 > 100 and value2 < 20')
        time2 = time.time() - start
        results["Query method"] = time2
        
        # Method 3: Using numpy where (for simple conditions)
        start = time.time()
        mask = (self.df['value1'].values > 100) & (self.df['value2'].values < 20)
        result3 = self.df[mask]
        time3 = time.time() - start
        results["Numpy arrays"] = time3
        
        # Verify results are the same
        assert len(result1) == len(result2) == len(result3), "Filtering results mismatch!"
        
        return {"Filtering": results}
    
    def _benchmark_groupby(self) -> Dict:
        """Benchmark different groupby operations"""
        print("\n2. GROUPBY OPERATIONS:")
        
        results = {}
        
        # Method 1: Basic groupby
        start = time.time()
        result1 = self.df.groupby('category').agg({'value1': 'mean', 'value2': 'sum'})
        time1 = time.time() - start
        results["Basic groupby"] = time1
        
        # Method 2: Optimized groupby with named aggregation
        start = time.time()
        result2 = self.df.groupby('category').agg(
            mean_value1=('value1', 'mean'),
            sum_value2=('value2', 'sum')
        )
        time2 = time.time() - start
        results["Named aggregation"] = time2
        
        # Method 3: Using value_counts for categorical aggregation
        start = time.time()
        category_counts = self.df['category'].value_counts()
        time3 = time.time() - start
        results["Value_counts"] = time3
        
        return {"GroupBy": results}
    
    def _benchmark_apply(self) -> Dict:
        """Benchmark apply vs vectorized operations"""
        print("\n3. APPLY VS VECTORIZED OPERATIONS:")
        
        results = {}
        
        # Method 1: Using apply (slow)
        start = time.time()
        result1 = self.df['value1'].apply(lambda x: x * 2 if x > 50 else x / 2)
        time1 = time.time() - start
        results["Apply method"] = time1
        
        # Method 2: Vectorized operations (fast)
        start = time.time()
        result2 = np.where(self.df['value1'] > 50, self.df['value1'] * 2, self.df['value1'] / 2)
        time2 = time.time() - start
        results["Vectorized (np.where)"] = time2
        
        # Method 3: Using pandas built-in (fastest)
        start = time.time()
        result3 = self.df['value1'] * 2
        mask = self.df['value1'] <= 50
        result3[mask] = self.df['value1'][mask] / 2
        time3 = time.time() - start
        results["Pandas built-in"] = time3
        
        return {"Apply Operations": results}
    
    def _benchmark_memory(self) -> Dict:
        """Benchmark memory optimization techniques"""
        print("\n4. MEMORY USAGE OPTIMIZATION:")
        
        results = {}
        
        # Original memory usage
        original_memory = self.df.memory_usage(deep=True).sum() / 1024**2  # MB
        
        # Method 1: Downcasting numeric types
        start = time.time()
        df_optimized = self.df.copy()
        
        # Downcast integers
        int_cols = df_optimized.select_dtypes(include=['int']).columns
        for col in int_cols:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
        
        # Downcast floats
        float_cols = df_optimized.select_dtypes(include=['float']).columns
        for col in float_cols:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        optimized_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2
        time1 = time.time() - start
        results["Downcasting"] = time1
        
        print(f"Memory reduction: {original_memory:.1f}MB -> {optimized_memory:.1f}MB")
        
        # Method 2: Categorical conversion
        start = time.time()
        df_categorical = self.df.copy()
        df_categorical['category'] = df_categorical['category'].astype('category')
        categorical_memory = df_categorical.memory_usage(deep=True).sum() / 1024**2
        time2 = time.time() - start
        results["Categorical conversion"] = time2
        
        print(f"With categorical: {categorical_memory:.1f}MB")
        
        return {"Memory Optimization": results}
    
    def _benchmark_datatypes(self) -> Dict:
        """Benchmark operations with different data types"""
        print("\n5. DATATYPE PERFORMANCE:")
        
        results = {}
        
        # String vs Categorical
        string_df = self.df.copy()
        string_df['category'] = string_df['category'].astype(str)
        
        start = time.time()
        string_result = string_df.groupby('category')['value1'].mean()
        time1 = time.time() - start
        results["String grouping"] = time1
        
        start = time.time()
        categorical_result = self.df.groupby('category')['value1'].mean()
        time2 = time.time() - start
        results["Categorical grouping"] = time2
        
        return {"Datatype Performance": results}
    
    def advanced_optimizations(self):
        """Show advanced optimization techniques"""
        print("\n" + "="*50)
        print("ADVANCED OPTIMIZATION TECHNIQUES")
        print("="*50)
        
        # 1. Chunk processing for very large datasets
        print("\n1. Chunk Processing Example:")
        chunk_size = 100000
        chunks_processed = 0
        
        for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
            # Process each chunk
            result = chunk.groupby('category').size()
            chunks_processed += 1
            if chunks_processed >= 3:  # Limit for demo
                break
        
        print(f"Processed {chunks_processed} chunks")
        
        # 2. Using Dask for parallel processing
        print("\n2. Parallel Processing (conceptual):")
        print("For datasets that don't fit in memory, consider:")
        print("  - Dask DataFrames")
        print("  - Modin")
        print("  - Vaex")
        print("  - Spark with PySpark")
        
        # 3. Efficient joining strategies
        print("\n3. Efficient Joining:")
        small_df = pd.DataFrame({
            'category': ['A', 'B', 'C', 'D', 'E'],
            'category_name': ['Alpha', 'Beta', 'Charlie', 'Delta', 'Echo']
        })
        
        # Efficient join with categorical
        self.df['category'] = self.df['category'].astype('category')
        small_df['category'] = small_df['category'].astype('category')
        
        start = time.time()
        joined = self.df.merge(small_df, on='category', how='left')
        join_time = time.time() - start
        
        print(f"Join time with categorical: {join_time:.4f}s")
        
        # 4. Using eval for complex expressions
        print("\n4. Using eval() for complex operations:")
        expr = "(value1 > 100) & (value2 < 20) & (value3 % 2 == 0)"
        
        start = time.time()
        result = self.df.eval(expr)
        eval_time = time.time() - start
        print(f"eval() time: {eval_time:.4f}s")

def performance_tips():
    """Provide performance optimization tips"""
    print("\n" + "="*50)
    print("PERFORMANCE OPTIMIZATION TIPS")
    print("="*50)
    
    tips = [
        "1. Use vectorized operations instead of apply()",
        "2. Convert object dtypes to category for low-cardinality columns",
        "3. Use appropriate numeric types (int8, float32, etc.)",
        "4. Use query() for complex filtering conditions",
        "5. Avoid chained indexing (use .loc[] instead)",
        "6. Use isin() instead of multiple OR conditions",
        "7. Prefer merge() over concat() for combining DataFrames",
        "8. Use method chaining to avoid intermediate variables",
        "9. Process data in chunks for very large datasets",
        "10. Consider using Dask/Modin for out-of-core processing"
    ]
    
    for tip in tips:
        print(tip)

if __name__ == "__main__":
    optimizer = PandasOptimizer(size=500000)  # Smaller size for demo
    optimizer.benchmark_operations()
    optimizer.advanced_optimizations()
    performance_tips()