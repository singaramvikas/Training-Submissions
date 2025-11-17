import pandas as pd
import numpy as np
import sqlite3
import time
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class SQLPerformanceOptimizer:
    """Optimize SQL joins and aggregations for performance"""
    
    def __init__(self, db_path=':memory:'):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute('PRAGMA journal_mode=WAL;')  # Write-Ahead Logging
        self.conn.execute('PRAGMA synchronous=NORMAL;')
        self.conn.execute('PRAGMA cache_size=-64000;')  # 64MB cache
        
    def create_large_dataset(self, scale_factor=1000):
        """Create a large dataset for performance testing"""
        print("Creating large dataset...")
        
        # Large customers table
        n_customers = 10000 * scale_factor
        customers_data = {
            'customer_id': range(1, n_customers + 1),
            'name': [f'Customer_{i}' for i in range(1, n_customers + 1)],
            'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], n_customers),
            'segment': np.random.choice(['Premium', 'Standard', 'Basic'], n_customers, p=[0.2, 0.5, 0.3])
        }
        customers_df = pd.DataFrame(customers_data)
        customers_df.to_sql('customers', self.conn, index=False, if_exists='replace')
        
        # Large orders table
        n_orders = 100000 * scale_factor
        orders_data = {
            'order_id': range(1, n_orders + 1),
            'customer_id': np.random.randint(1, n_customers + 1, n_orders),
            'order_date': pd.date_range('2023-01-01', periods=n_orders, freq='T'),
            'status': np.random.choice(['completed', 'pending', 'cancelled'], n_orders, p=[0.8, 0.15, 0.05]),
            'total_amount': np.random.exponential(100, n_orders)
        }
        orders_df = pd.DataFrame(orders_data)
        orders_df.to_sql('orders', self.conn, index=False, if_exists='replace')
        
        # Create indexes for performance
        self._create_indexes()
        
        print(f"Dataset created: {n_customers:,} customers, {n_orders:,} orders")
        return customers_df, orders_df
    
    def _create_indexes(self):
        """Create optimal indexes for query performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)",
            "CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date)",
            "CREATE INDEX IF NOT EXISTS idx_customers_city ON customers(city)",
            "CREATE INDEX IF NOT EXISTS idx_customers_segment ON customers(segment)"
        ]
        
        for index_sql in indexes:
            self.conn.execute(index_sql)
    
    def benchmark_queries(self):
        """Benchmark different query optimization techniques"""
        queries = {
            'naive_join': """
                SELECT c.customer_id, c.name, COUNT(o.order_id) as order_count
                FROM customers c
                JOIN orders o ON c.customer_id = o.customer_id
                WHERE o.status = 'completed'
                GROUP BY c.customer_id, c.name
            """,
            
            'optimized_join': """
                SELECT c.customer_id, c.name, o.order_count
                FROM customers c
                JOIN (
                    SELECT customer_id, COUNT(*) as order_count
                    FROM orders 
                    WHERE status = 'completed'
                    GROUP BY customer_id
                ) o ON c.customer_id = o.customer_id
            """,
            
            'window_function_naive': """
                SELECT customer_id, order_date, total_amount,
                       AVG(total_amount) OVER (PARTITION BY customer_id) as avg_customer_amount
                FROM orders
                WHERE status = 'completed'
            """,
            
            'window_function_optimized': """
                WITH customer_stats AS (
                    SELECT customer_id, AVG(total_amount) as avg_amount
                    FROM orders 
                    WHERE status = 'completed'
                    GROUP BY customer_id
                )
                SELECT o.customer_id, o.order_date, o.total_amount, cs.avg_amount as avg_customer_amount
                FROM orders o
                JOIN customer_stats cs ON o.customer_id = cs.customer_id
                WHERE o.status = 'completed'
            """,
            
            'complex_aggregation_naive': """
                SELECT 
                    c.city,
                    c.segment,
                    COUNT(DISTINCT o.order_id) as order_count,
                    SUM(o.total_amount) as total_revenue,
                    AVG(o.total_amount) as avg_order_value
                FROM customers c
                JOIN orders o ON c.customer_id = o.customer_id
                WHERE o.status = 'completed'
                GROUP BY c.city, c.segment
                ORDER BY total_revenue DESC
            """,
            
            'complex_aggregation_optimized': """
                WITH order_aggregates AS (
                    SELECT 
                        customer_id,
                        COUNT(DISTINCT order_id) as order_count,
                        SUM(total_amount) as total_revenue,
                        AVG(total_amount) as avg_order_value
                    FROM orders
                    WHERE status = 'completed'
                    GROUP BY customer_id
                )
                SELECT 
                    c.city,
                    c.segment,
                    SUM(oa.order_count) as order_count,
                    SUM(oa.total_revenue) as total_revenue,
                    AVG(oa.avg_order_value) as avg_order_value
                FROM customers c
                JOIN order_aggregates oa ON c.customer_id = oa.customer_id
                GROUP BY c.city, c.segment
                ORDER BY total_revenue DESC
            """
        }
        
        results = {}
        for query_name, query_sql in queries.items():
            start_time = time.time()
            
            # Execute query
            try:
                df = pd.read_sql_query(query_sql, self.conn)
                execution_time = time.time() - start_time
                results[query_name] = {
                    'execution_time': execution_time,
                    'row_count': len(df),
                    'query_type': query_name,
                    'optimized': 'optimized' in query_name
                }
                print(f"{query_name:30} | Time: {execution_time:6.2f}s | Rows: {len(df):8,}")
            except Exception as e:
                print(f"{query_name:30} | ERROR: {str(e)}")
                results[query_name] = {'error': str(e)}
        
        return results
    
    def analyze_query_plans(self):
        """Analyze and compare query execution plans"""
        queries = [
            "SELECT c.customer_id, COUNT(o.order_id) FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_id",
            "SELECT customer_id, COUNT(order_id) FROM orders GROUP BY customer_id"
        ]
        
        print("\n=== QUERY EXECUTION PLANS ===")
        for i, query in enumerate(queries):
            print(f"\nQuery {i+1}: {query}")
            plan = self.conn.execute(f"EXPLAIN QUERY PLAN {query}").fetchall()
            for step in plan:
                print(f"  {step}")
    
    def pandas_optimization_techniques(self):
        """Show pandas optimization techniques for large datasets"""
        print("\n=== PANDAS OPTIMIZATION TECHNIQUES ===")
        
        # Load sample data
        customers_df = pd.read_sql("SELECT * FROM customers LIMIT 100000", self.conn)
        orders_df = pd.read_sql("SELECT * FROM orders LIMIT 1000000", self.conn)
        
        # Technique 1: Using efficient data types
        print("\n1. Memory Usage with Different Data Types:")
        print(f"   Original memory: {customers_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Optimize data types
        customers_optimized = customers_df.copy()
        customers_optimized['customer_id'] = customers_optimized['customer_id'].astype('int32')
        customers_optimized['city'] = customers_optimized['city'].astype('category')
        customers_optimized['segment'] = customers_optimized['segment'].astype('category')
        print(f"   Optimized memory: {customers_optimized.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Technique 2: Efficient joins
        print("\n2. Join Performance:")
        
        # Method 1: Direct merge
        start_time = time.time()
        result1 = customers_df.merge(orders_df, on='customer_id')
        time1 = time.time() - start_time
        print(f"   Direct merge: {time1:.2f}s")
        
        # Method 2: Using indexes
        start_time = time.time()
        customers_indexed = customers_df.set_index('customer_id')
        orders_indexed = orders_df.set_index('customer_id')
        result2 = customers_indexed.join(orders_indexed, how='inner')
        time2 = time.time() - start_time
        print(f"   Index-based join: {time2:.2f}s")
        
        # Technique 3: Chunk processing for very large datasets
        print("\n3. Chunk Processing Example:")
        chunk_size = 10000
        chunks_processed = 0
        
        for chunk in pd.read_sql("SELECT * FROM orders", self.conn, chunksize=chunk_size):
            # Process each chunk
            chunk_agg = chunk.groupby('customer_id')['total_amount'].sum()
            chunks_processed += 1
            if chunks_processed >= 3:  # Limit for demo
                break
        
        print(f"   Processed {chunks_processed} chunks of {chunk_size} rows each")
        
        return {
            'memory_savings': customers_df.memory_usage().sum() / customers_optimized.memory_usage().sum(),
            'join_improvement': time1 / time2 if time2 > 0 else 0
        }
    
    def generate_performance_report(self):
        """Generate comprehensive performance optimization report"""
        print("=== SQL JOINS & AGGREGATIONS PERFORMANCE OPTIMIZATION ===\n")
        
        # Create large dataset
        self.create_large_dataset(scale_factor=10)
        
        # Benchmark queries
        print("1. QUERY PERFORMANCE BENCHMARK:")
        benchmark_results = self.benchmark_queries()
        
        # Analyze query plans
        self.analyze_query_plans()
        
        # Pandas optimizations
        pandas_results = self.pandas_optimization_techniques()
        
        # Performance recommendations
        print("\n=== PERFORMANCE OPTIMIZATION RECOMMENDATIONS ===")
        print("1. Use CTEs and subqueries to break down complex operations")
        print("2. Create appropriate indexes on join and filter columns")
        print("3. Use efficient data types in pandas (category, int32, etc.)")
        print("4. Process large datasets in chunks")
        print("5. Use EXPLAIN QUERY PLAN to analyze execution paths")
        print("6. Consider materialized views for frequently used aggregations")
        print("7. Use window functions judiciously - they can be expensive")
        
        return {
            'benchmark_results': benchmark_results,
            'pandas_optimizations': pandas_results
        }

# Execute performance analysis
if __name__ == "__main__":
    optimizer = SQLPerformanceOptimizer()
    report = optimizer.generate_performance_report()
    
    # Calculate overall performance improvements
    benchmark_results = report['benchmark_results']
    optimized_times = [r['execution_time'] for r in benchmark_results.values() 
                      if 'execution_time' in r and 'optimized' in r and r['optimized']]
    naive_times = [r['execution_time'] for r in benchmark_results.values() 
                  if 'execution_time' in r and 'optimized' in r and not r['optimized']]
    
    if naive_times and optimized_times:
        avg_improvement = sum(naive_times) / sum(optimized_times)
        print(f"\nOverall performance improvement with optimizations: {avg_improvement:.1f}x faster")