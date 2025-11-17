import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

class SQLAdvancedAnalysis:
    def __init__(self):
        self.conn = sqlite3.connect(':memory:')
        
    def create_sample_dataset(self):
        """Create a realistic e-commerce dataset"""
        # Customers table
        customers_data = {
            'customer_id': range(1, 101),
            'name': [f'Customer_{i}' for i in range(1, 101)],
            'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'] * 20,
            'signup_date': pd.date_range('2023-01-01', periods=100, freq='D')
        }
        customers_df = pd.DataFrame(customers_data)
        
        # Products table
        products_data = {
            'product_id': range(1, 21),
            'product_name': [f'Product_{i}' for i in range(1, 21)],
            'category': ['Electronics', 'Clothing', 'Home', 'Books'] * 5,
            'price': [i * 10 + 50 for i in range(20)]
        }
        products_df = pd.DataFrame(products_data)
        
        # Orders table
        orders_data = {
            'order_id': range(1, 501),
            'customer_id': [i % 100 + 1 for i in range(500)],
            'order_date': pd.date_range('2024-01-01', periods=500, freq='H'),
            'status': ['completed', 'pending', 'cancelled'] * 166 + ['completed', 'pending']
        }
        orders_df = pd.DataFrame(orders_data)
        
        # Order items table
        order_items_data = {
            'order_item_id': range(1, 1201),
            'order_id': [i % 500 + 1 for i in range(1200)],
            'product_id': [i % 20 + 1 for i in range(1200)],
            'quantity': [i % 5 + 1 for i in range(1200)],
            'unit_price': [products_data['price'][i % 20] for i in range(1200)]
        }
        order_items_df = pd.DataFrame(order_items_data)
        
        # Save to SQLite
        customers_df.to_sql('customers', self.conn, index=False, if_exists='replace')
        products_df.to_sql('products', self.conn, index=False, if_exists='replace')
        orders_df.to_sql('orders', self.conn, index=False, if_exists='replace')
        order_items_df.to_sql('order_items', self.conn, index=False, if_exists='replace')
        
        return customers_df, products_df, orders_df, order_items_df
    
    def advanced_joins_analysis(self):
        """Perform advanced joins and aggregations"""
        queries = {
            'customer_purchase_analysis': """
                SELECT 
                    c.customer_id,
                    c.name,
                    c.city,
                    COUNT(DISTINCT o.order_id) as total_orders,
                    SUM(oi.quantity * oi.unit_price) as total_spent,
                    AVG(oi.quantity * oi.unit_price) as avg_order_value,
                    MAX(o.order_date) as last_order_date
                FROM customers c
                LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
                LEFT JOIN order_items oi ON o.order_id = oi.order_id
                GROUP BY c.customer_id, c.name, c.city
                ORDER BY total_spent DESC
            """,
            
            'product_performance_by_category': """
                SELECT 
                    p.category,
                    p.product_name,
                    COUNT(oi.order_item_id) as times_ordered,
                    SUM(oi.quantity) as total_quantity,
                    SUM(oi.quantity * oi.unit_price) as total_revenue,
                    RANK() OVER (PARTITION BY p.category ORDER BY SUM(oi.quantity * oi.unit_price) DESC) as revenue_rank
                FROM products p
                LEFT JOIN order_items oi ON p.product_id = oi.product_id
                LEFT JOIN orders o ON oi.order_id = o.order_id AND o.status = 'completed'
                GROUP BY p.category, p.product_name
                ORDER BY p.category, revenue_rank
            """,
            
            'monthly_sales_trends': """
                SELECT 
                    strftime('%Y-%m', o.order_date) as order_month,
                    p.category,
                    COUNT(DISTINCT o.order_id) as order_count,
                    SUM(oi.quantity) as total_quantity,
                    SUM(oi.quantity * oi.unit_price) as monthly_revenue,
                    LAG(SUM(oi.quantity * oi.unit_price)) OVER (PARTITION BY p.category ORDER BY strftime('%Y-%m', o.order_date)) as prev_month_revenue
                FROM orders o
                JOIN order_items oi ON o.order_id = oi.order_id
                JOIN products p ON oi.product_id = p.product_id
                WHERE o.status = 'completed'
                GROUP BY order_month, p.category
                ORDER BY order_month, p.category
            """
        }
        
        results = {}
        for name, query in queries.items():
            results[name] = pd.read_sql_query(query, self.conn)
            
        return results
    
    def analyze_and_visualize(self):
        """Analyze results and create visualizations"""
        results = self.advanced_joins_analysis()
        
        # Customer analysis
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        customer_stats = results['customer_purchase_analysis']
        top_customers = customer_stats.nlargest(10, 'total_spent')
        plt.barh(top_customers['name'], top_customers['total_spent'])
        plt.title('Top 10 Customers by Total Spending')
        plt.xlabel('Total Spent ($)')
        
        plt.subplot(2, 2, 2)
        category_revenue = results['product_performance_by_category'].groupby('category')['total_revenue'].sum()
        plt.pie(category_revenue, labels=category_revenue.index, autopct='%1.1f%%')
        plt.title('Revenue Distribution by Category')
        
        plt.subplot(2, 2, 3)
        monthly_trends = results['monthly_sales_trends']
        pivot_trends = monthly_trends.pivot(index='order_month', columns='category', values='monthly_revenue')
        pivot_trends.plot(kind='line', ax=plt.gca())
        plt.title('Monthly Revenue Trends by Category')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        city_spending = customer_stats.groupby('city')['total_spent'].sum()
        plt.bar(city_spending.index, city_spending.values)
        plt.title('Total Spending by City')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('advanced_sql_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results

# Execute analysis
if __name__ == "__main__":
    analyzer = SQLAdvancedAnalysis()
    analyzer.create_sample_dataset()
    results = analyzer.analyze_and_visualize()
    
    # Print key insights
    print("=== ADVANCED SQL JOINS & AGGREGATIONS ANALYSIS ===")
    print("\n1. Customer Insights:")
    customer_stats = results['customer_purchase_analysis']
    print(f"   - Total Customers: {len(customer_stats)}")
    print(f"   - Average Customer Spending: ${customer_stats['total_spent'].mean():.2f}")
    print(f"   - Top Spender: {customer_stats.iloc[0]['name']} (${customer_stats.iloc[0]['total_spent']:.2f})")
    
    print("\n2. Product Performance:")
    product_stats = results['product_performance_by_category']
    top_products = product_stats[product_stats['revenue_rank'] == 1]
    for _, product in top_products.iterrows():
        print(f"   - {product['category']}: {product['product_name']} (${product['total_revenue']:.2f})")