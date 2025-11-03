import numpy as np
import pandas as pd
from numba import jit
from concurrent.futures import ThreadPoolExecutor
import time
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

@jit(nopython=True)
def fast_correlation(x, y):
    """Optimized correlation calculation using Numba"""
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
    sum_y2 = np.sum(y * y)
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
    
    return numerator / denominator if denominator != 0 else 0

class OptimizedDataProcessor:
    """Optimized data processing class with performance enhancements"""
    
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs if n_jobs != -1 else None
    
    def parallel_feature_engineering(self, X):
        """Parallel feature engineering for better performance"""
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            
            # Create multiple feature engineering tasks
            futures.append(executor.submit(self._create_statistical_features, X))
            futures.append(executor.submit(self._create_interaction_features, X))
            futures.append(executor.submit(self._create_polynomial_features, X))
            
            # Collect results
            results = [future.result() for future in futures]
        
        # Combine all features
        return np.hstack(results)
    
    def _create_statistical_features(self, X):
        """Create statistical features"""
        return np.column_stack([
            np.mean(X, axis=1).reshape(-1, 1),
            np.std(X, axis=1).reshape(-1, 1),
            np.min(X, axis=1).reshape(-1, 1),
            np.max(X, axis=1).reshape(-1, 1)
        ])
    
    def _create_interaction_features(self, X):
        """Create interaction features"""
        n_samples, n_features = X.shape
        interactions = []
        
        for i in range(min(5, n_features)):  # Limit to avoid combinatorial explosion
            for j in range(i+1, min(10, n_features)):
                interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
        
        return np.hstack(interactions) if interactions else np.empty((n_samples, 0))
    
    def _create_polynomial_features(self, X):
        """Create polynomial features"""
        return np.column_stack([X[:, :5]**2, X[:, :5]**3])  # Limit to first 5 features

def benchmark_optimization():
    """Benchmark performance improvements"""
    
    # Generate large dataset
    X, y = make_classification(
        n_samples=10000, 
        n_features=100, 
        n_informative=50,
        n_redundant=30,
        random_state=42
    )
    
    print("=== Performance Optimization Benchmark ===")
    print(f"Dataset size: {X.shape}")
    
    # Benchmark original approach
    start_time = time.time()
    
    # Original processing (naive approach)
    processor = OptimizedDataProcessor(n_jobs=4)
    X_enhanced = processor.parallel_feature_engineering(X)
    
    original_time = time.time() - start_time
    
    # Benchmark optimized model training
    start_time = time.time()
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        n_jobs=4,
        random_state=42
    )
    
    # Use cross-validation for robust evaluation
    cv_scores = cross_val_score(model, X_enhanced, y, cv=3, n_jobs=2)
    
    optimized_time = time.time() - start_time
    
    print(f"\n=== Performance Results ===")
    print(f"Enhanced features shape: {X_enhanced.shape}")
    print(f"Feature engineering time: {original_time:.4f} seconds")
    print(f"Model training + evaluation time: {optimized_time:.4f} seconds")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Memory optimization demonstration
    print(f"\n=== Memory Optimization ===")
    print(f"Original data type: {X.dtype}")
    print(f"Original memory usage: {X.nbytes / 1024 / 1024:.2f} MB")
    
    # Convert to float32 for memory savings
    X_optimized = X.astype(np.float32)
    print(f"Optimized data type: {X_optimized.dtype}")
    print(f"Optimized memory usage: {X_optimized.nbytes / 1024 / 1024:.2f} MB")
    print(f"Memory reduction: {(1 - X_optimized.nbytes / X.nbytes) * 100:.1f}%")

# Run optimization benchmark
benchmark_optimization()