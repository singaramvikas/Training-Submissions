"""
Hard 1: Optimize the implementation of Python Data Structures for performance.
"""

import time
import random
import string
from collections import defaultdict, deque, Counter
import numpy as np
import pandas as pd
from functools import lru_cache
import heapq
from dataclasses import dataclass
from typing import List, Dict, Set
import memory_profiler

class PerformanceOptimizer:
    """Demonstrate optimized data structures for performance"""
    
    def __init__(self, data_size=10000):
        self.data_size = data_size
        self.results = {}
    
    def generate_test_data(self):
        """Generate large test dataset"""
        print("Generating test data...")
        
        # Large list of random strings
        self.strings = [''.join(random.choices(string.ascii_letters, k=10)) 
                       for _ in range(self.data_size)]
        
        # Large list of numbers
        self.numbers = [random.randint(1, 1000) for _ in range(self.data_size)]
        
        # Key-value pairs
        self.keys = [f"key_{i}" for i in range(self.data_size)]
        self.values = [random.random() for _ in range(self.data_size)]
    
    def benchmark_list_operations(self):
        """Benchmark and optimize list operations"""
        print("\nLIST OPERATIONS BENCHMARK")
        print("=" * 30)
        
        # Test data
        test_list = self.numbers.copy()
        
        # Naive approach - repeated appends in loop
        start_time = time.time()
        result_naive = []
        for num in test_list:
            result_naive.append(num * 2)
        naive_time = time.time() - start_time
        
        # Optimized approach - list comprehension
        start_time = time.time()
        result_optimized = [num * 2 for num in test_list]
        optimized_time = time.time() - start_time
        
        # Most optimized - numpy arrays
        start_time = time.time()
        np_array = np.array(test_list)
        result_numpy = (np_array * 2).tolist()
        numpy_time = time.time() - start_time
        
        print(f"Naive (for loop): {naive_time:.6f} seconds")
        print(f"Optimized (list comprehension): {optimized_time:.6f} seconds")
        print(f"Most Optimized (numpy): {numpy_time:.6f} seconds")
        print(f"Speedup: {naive_time/optimized_time:.2f}x")
        
        self.results['list_operations'] = {
            'naive': naive_time,
            'optimized': optimized_time,
            'numpy': numpy_time,
            'speedup': naive_time / optimized_time
        }
    
    def benchmark_membership_testing(self):
        """Benchmark membership testing in different structures"""
        print("\nMEMBERSHIP TESTING BENCHMARK")
        print("=" * 35)
        
        test_data = self.strings.copy()
        search_items = random.sample(test_data, 100) + ['nonexistent'] * 10
        
        # List membership test (O(n))
        start_time = time.time()
        list_results = []
        for item in search_items:
            list_results.append(item in test_data)
        list_time = time.time() - start_time
        
        # Set membership test (O(1))
        test_set = set(test_data)
        start_time = time.time()
        set_results = []
        for item in search_items:
            set_results.append(item in test_set)
        set_time = time.time() - start_time
        
        print(f"List membership: {list_time:.6f} seconds")
        print(f"Set membership: {set_time:.6f} seconds")
        print(f"Speedup: {list_time/set_time:.2f}x")
        
        self.results['membership_testing'] = {
            'list': list_time,
            'set': set_time,
            'speedup': list_time / set_time
        }
    
    def benchmark_dictionary_operations(self):
        """Benchmark dictionary operations and optimizations"""
        print("\nDICTIONARY OPERATIONS BENCHMARK")
        print("=" * 40)
        
        # Naive dictionary creation
        start_time = time.time()
        naive_dict = {}
        for key, value in zip(self.keys, self.values):
            naive_dict[key] = value
        naive_time = time.time() - start_time
        
        # Optimized dictionary creation
        start_time = time.time()
        optimized_dict = dict(zip(self.keys, self.values))
        optimized_time = time.time() - start_time
        
        # Dictionary comprehension
        start_time = time.time()
        comprehension_dict = {k: v for k, v in zip(self.keys, self.values)}
        comprehension_time = time.time() - start_time
        
        print(f"Naive (loop): {naive_time:.6f} seconds")
        print(f"Optimized (dict zip): {optimized_time:.6f} seconds")
        print(f"Comprehension: {comprehension_time:.6f} seconds")
        
        # Lookup performance
        test_keys = random.sample(self.keys, 1000)
        
        start_time = time.time()
        for key in test_keys:
            _ = naive_dict[key]
        lookup_time = time.time() - start_time
        
        print(f"Lookup 1000 keys: {lookup_time:.6f} seconds")
        
        self.results['dictionary_operations'] = {
            'naive_creation': naive_time,
            'optimized_creation': optimized_time,
            'comprehension_creation': comprehension_time,
            'lookup_time': lookup_time
        }
    
    def benchmark_data_aggregation(self):
        """Benchmark data aggregation techniques"""
        print("\nDATA AGGREGATION BENCHMARK")
        print("=" * 30)
        
        # Sample data: product categories and sales
        categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']
        sales_data = [(random.choice(categories), random.randint(10, 1000)) 
                     for _ in range(self.data_size)]
        
        # Naive approach using default dictionary
        start_time = time.time()
        naive_aggregation = defaultdict(int)
        for category, amount in sales_data:
            naive_aggregation[category] += amount
        naive_time = time.time() - start_time
        
        # Optimized using Counter
        start_time = time.time()
        counter_aggregation = Counter()
        for category, amount in sales_data:
            counter_aggregation[category] += amount
        counter_time = time.time() - start_time
        
        # Using pandas (most optimized for large data)
        start_time = time.time()
        df = pd.DataFrame(sales_data, columns=['category', 'amount'])
        pandas_aggregation = df.groupby('category')['amount'].sum().to_dict()
        pandas_time = time.time() - start_time
        
        print(f"Naive (defaultdict): {naive_time:.6f} seconds")
        print(f"Optimized (Counter): {counter_time:.6f} seconds")
        print(f"Most Optimized (pandas): {pandas_time:.6f} seconds")
        
        self.results['data_aggregation'] = {
            'naive': naive_time,
            'counter': counter_time,
            'pandas': pandas_time
        }
    
    def demonstrate_memory_optimization(self):
        """Demonstrate memory optimization techniques"""
        print("\nMEMORY OPTIMIZATION TECHNIQUES")
        print("=" * 35)
        
        # Using __slots__ for memory efficiency
        class RegularClass:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
        
        @dataclass
        class SlotsClass:
            __slots__ = ['x', 'y', 'z']
            x: int
            y: float
            z: str
        
        # Create many instances
        num_instances = 10000
        
        # Memory usage for regular class
        regular_instances = [RegularClass(i, i*1.5, str(i)) 
                           for i in range(num_instances)]
        
        # Memory usage for slots class
        slots_instances = [SlotsClass(i, i*1.5, str(i)) 
                         for i in range(num_instances)]
        
        print("Memory optimization using __slots__:")
        print(f"Regular class instances created: {len(regular_instances)}")
        print(f"Slots class instances created: {len(slots_instances)}")
        print("Note: Use memory_profiler for precise memory measurements")
        
        self.results['memory_optimization'] = {
            'regular_instances': len(regular_instances),
            'slots_instances': len(slots_instances)
        }
    
    def advanced_optimization_techniques(self):
        """Show advanced optimization techniques"""
        print("\nADVANCED OPTIMIZATION TECHNIQUES")
        print("=" * 40)
        
        # 1. Using generators for memory efficiency
        def naive_range_sum(n):
            """Naive approach - creates full list in memory"""
            numbers = list(range(n))
            return sum(numbers)
        
        def optimized_range_sum(n):
            """Optimized approach - uses generator"""
            return sum(range(n))
        
        # 2. LRU caching for expensive function calls
        @lru_cache(maxsize=128)
        def expensive_function(x):
            time.sleep(0.001)  # Simulate expensive operation
            return x * x
        
        # Benchmark caching
        test_values = [1, 2, 3, 1, 2, 3, 4, 5]  # Repeated values
        
        start_time = time.time()
        without_cache = [expensive_function.__wrapped__(x) for x in test_values]
        time_without_cache = time.time() - start_time
        
        start_time = time.time()
        with_cache = [expensive_function(x) for x in test_values]
        time_with_cache = time.time() - start_time
        
        print(f"Without caching: {time_without_cache:.6f} seconds")
        print(f"With LRU caching: {time_with_cache:.6f} seconds")
        print(f"Caching speedup: {time_without_cache/time_with_cache:.2f}x")
        
        # 3. Using heapq for priority queue operations
        large_list = [random.randint(1, 10000) for _ in range(1000)]
        
        # Naive approach for getting top N
        start_time = time.time()
        naive_top = sorted(large_list, reverse=True)[:10]
        naive_top_time = time.time() - start_time
        
        # Optimized using heapq
        start_time = time.time()
        heapq_top = heapq.nlargest(10, large_list)
        heapq_top_time = time.time() - start_time
        
        print(f"\nTop 10 elements:")
        print(f"Naive (sort): {naive_top_time:.6f} seconds")
        print(f"Optimized (heapq): {heapq_top_time:.6f} seconds")
        
        self.results['advanced_techniques'] = {
            'caching_speedup': time_without_cache / time_with_cache,
            'heapq_speedup': naive_top_time / heapq_top_time
        }
    
    def run_all_benchmarks(self):
        """Run all performance benchmarks"""
        print("PERFORMANCE OPTIMIZATION DEMONSTRATION")
        print("=" * 45)
        print(f"Data size: {self.data_size} elements")
        
        self.generate_test_data()
        self.benchmark_list_operations()
        self.benchmark_membership_testing()
        self.benchmark_dictionary_operations()
        self.benchmark_data_aggregation()
        self.demonstrate_memory_optimization()
        self.advanced_optimization_techniques()
        
        self.print_summary()
    
    def print_summary(self):
        """Print summary of all optimization results"""
        print("\n" + "=" * 50)
        print("PERFORMANCE OPTIMIZATION SUMMARY")
        print("=" * 50)
        
        for test_name, results in self.results.items():
            print(f"\n{test_name.replace('_', ' ').title()}:")
            for metric, value in results.items():
                if 'speedup' in metric:
                    print(f"  {metric}: {value:.2f}x")
                elif 'time' in metric:
                    print(f"  {metric}: {value:.6f} seconds")
                else:
                    print(f"  {metric}: {value}")

def main():
    """Main execution function"""
    optimizer = PerformanceOptimizer(data_size=10000)
    optimizer.run_all_benchmarks()

if __name__ == "__main__":
    main()