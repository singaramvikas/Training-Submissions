"""
Intermediate 1: Apply Python Data Structures on a real dataset and explain results.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import json

class EcommerceDataAnalyzer:
    """Analyze e-commerce data using Python data structures"""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = None
        self.analysis_results = {}
        
    def load_data(self):
        """Load and prepare the dataset"""
        try:
            # Using a sample e-commerce dataset
            # You can replace this with any CSV file or use the provided sample data
            self.data = pd.read_csv(self.dataset_path)
            print(f"Dataset loaded: {len(self.data)} records")
            return True
        except FileNotFoundError:
            # Create sample data if file not found
            print("Creating sample e-commerce data...")
            self._create_sample_data()
            return True
    
    def _create_sample_data(self):
        """Create sample e-commerce data for demonstration"""
        np.random.seed(42)
        sample_size = 1000
        
        sample_data = {
            'order_id': range(1000, 1000 + sample_size),
            'customer_id': np.random.randint(100, 500, sample_size),
            'product_id': np.random.choice(['P001', 'P002', 'P003', 'P004', 'P005', 'P006'], sample_size),
            'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], sample_size),
            'price': np.random.uniform(10, 500, sample_size).round(2),
            'quantity': np.random.randint(1, 5, sample_size),
            'order_date': pd.date_range('2023-01-01', periods=sample_size, freq='H'),
            'customer_region': np.random.choice(['North', 'South', 'East', 'West'], sample_size)
        }
        
        self.data = pd.DataFrame(sample_data)
        self.data.to_csv('sample_ecommerce_data.csv', index=False)
    
    def analyze_with_native_structures(self):
        """Perform analysis using native Python data structures"""
        if self.data is None:
            self.load_data()
        
        # Convert to dictionary for analysis
        records = self.data.to_dict('records')
        
        # 1. Using Counter for product popularity
        product_counter = Counter(record['product_id'] for record in records)
        self.analysis_results['product_popularity'] = dict(product_counter.most_common())
        
        # 2. Using defaultdict for category analysis
        category_sales = defaultdict(float)
        category_quantity = defaultdict(int)
        
        for record in records:
            category_sales[record['category']] += record['price'] * record['quantity']
            category_quantity[record['category']] += record['quantity']
        
        self.analysis_results['category_sales'] = dict(category_sales)
        self.analysis_results['category_quantity'] = dict(category_quantity)
        
        # 3. Using dictionary for customer analysis
        customer_orders = defaultdict(list)
        for record in records:
            customer_orders[record['customer_id']].append(record)
        
        customer_stats = {}
        for customer_id, orders in customer_orders.items():
            total_spent = sum(order['price'] * order['quantity'] for order in orders)
            customer_stats[customer_id] = {
                'total_orders': len(orders),
                'total_spent': total_spent,
                'average_order_value': total_spent / len(orders)
            }
        
        self.analysis_results['customer_stats'] = customer_stats
        
        # 4. Using set for unique analysis
        unique_customers = set(record['customer_id'] for record in records)
        unique_products = set(record['product_id'] for record in records)
        
        self.analysis_results['unique_counts'] = {
            'customers': len(unique_customers),
            'products': len(unique_products)
        }
        
        # 5. Regional analysis using nested dictionaries
        regional_sales = defaultdict(lambda: defaultdict(float))
        for record in records:
            regional_sales[record['customer_region']][record['category']] += record['price'] * record['quantity']
        
        self.analysis_results['regional_sales'] = {region: dict(categories) 
                                                 for region, categories in regional_sales.items()}
    
    def explain_results(self):
        """Explain the analysis results"""
        print("E-COMMERCE DATA ANALYSIS RESULTS")
        print("=" * 40)
        
        print(f"\n1. PRODUCT POPULARITY (Using Counter):")
        for product, count in list(self.analysis_results['product_popularity'].items())[:5]:
            print(f"   {product}: {count} orders")
        
        print(f"\n2. CATEGORY SALES (Using defaultdict):")
        for category, sales in self.analysis_results['category_sales'].items():
            print(f"   {category}: ${sales:,.2f}")
        
        print(f"\n3. CUSTOMER STATISTICS (Using dictionary):")
        top_customers = sorted(self.analysis_results['customer_stats'].items(), 
                             key=lambda x: x[1]['total_spent'], reverse=True)[:3]
        for customer, stats in top_customers:
            print(f"   Customer {customer}: {stats['total_orders']} orders, "
                  f"${stats['total_spent']:.2f} total spent")
        
        print(f"\n4. UNIQUE COUNTS (Using set):")
        print(f"   Unique Customers: {self.analysis_results['unique_counts']['customers']}")
        print(f"   Unique Products: {self.analysis_results['unique_counts']['products']}")
        
        print(f"\n5. REGIONAL SALES (Using nested defaultdict):")
        for region, categories in self.analysis_results['regional_sales'].items():
            top_category = max(categories.items(), key=lambda x: x[1])
            print(f"   {region}: Top category - {top_category[0]} (${top_category[1]:,.2f})")
    
    def save_analysis(self, filename='analysis_results.json'):
        """Save analysis results to JSON"""
        with open(filename, 'w') as f:
            # Convert defaultdict to regular dict for JSON serialization
            json_ready = {}
            for key, value in self.analysis_results.items():
                if isinstance(value, defaultdict):
                    json_ready[key] = dict(value)
                else:
                    json_ready[key] = value
            json.dump(json_ready, f, indent=2, default=str)
        print(f"\nAnalysis saved to {filename}")

def main():
    """Main execution function"""
    analyzer = EcommerceDataAnalyzer('ecommerce_data.csv')  # Replace with your dataset
    analyzer.load_data()
    analyzer.analyze_with_native_structures()
    analyzer.explain_results()
    analyzer.save_analysis()

if __name__ == "__main__":
    main()