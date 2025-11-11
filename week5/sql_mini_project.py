"""
Hard 2: Build a mini project applying SQL Basics end-to-end.

E-Commerce Analytics System
A complete system that demonstrates SQL operations on an e-commerce dataset.
"""

import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class ECommerceAnalytics:
    """
    End-to-end e-commerce analytics system using SQL-like operations.
    """
    
    def __init__(self):
        self.conn = sqlite3.connect(':memory:')  # In-memory SQLite database
        self.cursor = self.conn.cursor()
        self.setup_database()
    
    def setup_database(self):
        """Create and populate the e-commerce database"""
        
        # Create tables
        self.cursor.execute('''
            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                country TEXT,
                signup_date DATE
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE products (
                product_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT,
                price DECIMAL(10,2),
                stock_quantity INTEGER
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                order_date DATE,
                total_amount DECIMAL(10,2),
                status TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE order_items (
                order_item_id INTEGER PRIMARY KEY,
                order_id INTEGER,
                product_id INTEGER,
                quantity INTEGER,
                unit_price DECIMAL(10,2),
                FOREIGN KEY (order_id) REFERENCES orders (order_id),
                FOREIGN KEY (product_id) REFERENCES products (product_id)
            )
        ''')
        
        # Insert sample data
        self._insert_sample_data()
        print("E-commerce database setup complete!")
    
    def _insert_sample_data(self):
        """Insert sample data into all tables"""
        
        # Customers
        customers = [
            (1, 'John Smith', 'john@email.com', 'USA', '2023-01-15'),
            (2, 'Maria Garcia', 'maria@email.com', 'Spain', '2023-02-20'),
            (3, 'Chen Wei', 'chen@email.com', 'China', '2023-01-10'),
            (4, 'Sarah Johnson', 'sarah@email.com', 'UK', '2023-03-05'),
            (5, 'Ahmed Hassan', 'ahmed@email.com', 'Egypt', '2023-02-28')
        ]
        self.cursor.executemany('INSERT INTO customers VALUES (?,?,?,?,?)', customers)
        
        # Products
        products = [
            (1, 'Laptop', 'Electronics', 999.99, 50),
            (2, 'Smartphone', 'Electronics', 699.99, 100),
            (3, 'Desk Chair', 'Furniture', 149.99, 30),
            (4, 'Coffee Maker', 'Home Appliances', 79.99, 75),
            (5, 'Running Shoes', 'Sports', 119.99, 60),
            (6, 'Novel', 'Books', 14.99, 200),
            (7, 'Headphones', 'Electronics', 199.99, 40)
        ]
        self.cursor.executemany('INSERT INTO products VALUES (?,?,?,?,?)', products)
        
        # Orders
        orders = [
            (1, 1, '2023-03-01', 1119.98, 'delivered'),
            (2, 2, '2023-03-02', 819.98, 'processing'),
            (3, 3, '2023-03-02', 149.99, 'delivered'),
            (4, 1, '2023-03-03', 199.99, 'shipped'),
            (5, 4, '2023-03-04', 214.98, 'processing'),
            (6, 5, '2023-03-05', 699.99, 'delivered')
        ]
        self.cursor.executemany('INSERT INTO orders VALUES (?,?,?,?,?)', orders)
        
        # Order items
        order_items = [
            (1, 1, 1, 1, 999.99),
            (2, 1, 7, 1, 199.99),
            (3, 2, 2, 1, 699.99),
            (4, 2, 6, 1, 14.99),
            (5, 2, 6, 1, 14.99),
            (6, 3, 3, 1, 149.99),
            (7, 4, 7, 1, 199.99),
            (8, 5, 5, 1, 119.99),
            (9, 5, 6, 1, 14.99),
            (10, 6, 2, 1, 699.99)
        ]
        self.cursor.executemany('INSERT INTO order_items VALUES (?,?,?,?,?)', order_items)
        
        self.conn.commit()
    
    def execute_query(self, query, params=None):
        """Execute SQL query and return results as DataFrame"""
        if params:
            result = pd.read_sql_query(query, self.conn, params=params)
        else:
            result = pd.read_sql_query(query, self.conn)
        return result
    
    def sales_analysis(self):
        """Comprehensive sales analysis"""
        print("SALES ANALYSIS")
        print("=" * 40)
        
        # Total sales by category
        query = '''
            SELECT p.category, SUM(oi.quantity * oi.unit_price) as total_sales,
                   COUNT(DISTINCT o.order_id) as order_count
            FROM order_items oi
            JOIN products p ON oi.product_id = p.product_id
            JOIN orders o ON oi.order_id = o.order_id
            GROUP BY p.category
            ORDER BY total_sales DESC
        '''
        category_sales = self.execute_query(query)
        print("\n1. Sales by Category:")
        print(category_sales.to_string(index=False))
        
        # Monthly sales trend
        query = '''
            SELECT strftime('%Y-%m', order_date) as month,
                   SUM(total_amount) as monthly_sales,
                   COUNT(*) as order_count
            FROM orders
            GROUP BY month
            ORDER BY month
        '''
        monthly_trend = self.execute_query(query)
        print("\n2. Monthly Sales Trend:")
        print(monthly_trend.to_string(index=False))
        
        return category_sales, monthly_trend
    
    def customer_analysis(self):
        """Customer behavior analysis"""
        print("\nCUSTOMER ANALYSIS")
        print("=" * 40)
        
        # Top customers by spending
        query = '''
            SELECT c.name, c.country, 
                   SUM(o.total_amount) as total_spent,
                   COUNT(o.order_id) as order_count
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            GROUP BY c.customer_id
            ORDER BY total_spent DESC
        '''
        top_customers = self.execute_query(query)
        print("\n1. Top Customers by Spending:")
        print(top_customers.to_string(index=False))
        
        # Customer country distribution
        query = '''
            SELECT country, COUNT(*) as customer_count,
                   AVG(order_count) as avg_orders_per_customer
            FROM (
                SELECT c.country, c.customer_id, COUNT(o.order_id) as order_count
                FROM customers c
                LEFT JOIN orders o ON c.customer_id = o.customer_id
                GROUP BY c.customer_id
            )
            GROUP BY country
        '''
        country_analysis = self.execute_query(query)
        print("\n2. Customer Distribution by Country:")
        print(country_analysis.to_string(index=False))
        
        return top_customers, country_analysis
    
    def inventory_analysis(self):
        """Product and inventory analysis"""
        print("\nINVENTORY ANALYSIS")
        print("=" * 40)
        
        # Best selling products
        query = '''
            SELECT p.name, p.category, p.price,
                   SUM(oi.quantity) as total_quantity_sold,
                   SUM(oi.quantity * oi.unit_price) as total_revenue
            FROM products p
            JOIN order_items oi ON p.product_id = oi.product_id
            GROUP BY p.product_id
            ORDER BY total_revenue DESC
        '''
        best_sellers = self.execute_query(query)
        print("\n1. Best Selling Products:")
        print(best_sellers.to_string(index=False))
        
        # Stock analysis
        query = '''
            SELECT name, category, stock_quantity,
                   CASE 
                       WHEN stock_quantity < 20 THEN 'Low Stock'
                       WHEN stock_quantity < 50 THEN 'Medium Stock'
                       ELSE 'High Stock'
                   END as stock_status
            FROM products
            ORDER BY stock_quantity ASC
        '''
        stock_analysis = self.execute_query(query)
        print("\n2. Stock Level Analysis:")
        print(stock_analysis.to_string(index=False))
        
        return best_sellers, stock_analysis
    
    def business_metrics(self):
        """Key business performance metrics"""
        print("\nBUSINESS PERFORMANCE METRICS")
        print("=" * 40)
        
        metrics = {}
        
        # Total revenue
        query = "SELECT SUM(total_amount) as total_revenue FROM orders"
        metrics['total_revenue'] = self.execute_query(query).iloc[0, 0]
        
        # Average order value
        query = "SELECT AVG(total_amount) as avg_order_value FROM orders"
        metrics['avg_order_value'] = self.execute_query(query).iloc[0, 0]
        
        # Total customers
        query = "SELECT COUNT(*) as total_customers FROM customers"
        metrics['total_customers'] = self.execute_query(query).iloc[0, 0]
        
        # Orders per customer
        query = '''
            SELECT AVG(order_count) as avg_orders_per_customer 
            FROM (
                SELECT customer_id, COUNT(*) as order_count 
                FROM orders 
                GROUP BY customer_id
            )
        '''
        metrics['avg_orders_per_customer'] = self.execute_query(query).iloc[0, 0]
        
        # Print metrics
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric.replace('_', ' ').title()}: ${value:.2f}")
            else:
                print(f"{metric.replace('_', ' ').title()}: {value}")
        
        return metrics
    
    def generate_report(self):
        """Generate comprehensive business report"""
        print("E-COMMERCE ANALYTICS REPORT")
        print("=" * 50)
        print(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print()
        
        # Run all analyses
        sales_by_category, monthly_trend = self.sales_analysis()
        top_customers, country_analysis = self.customer_analysis()
        best_sellers, stock_analysis = self.inventory_analysis()
        metrics = self.business_metrics()
        
        # Create visualizations
        self.create_visualizations(sales_by_category, monthly_trend, top_customers)
        
        return {
            'sales_analysis': {
                'by_category': sales_by_category,
                'monthly_trend': monthly_trend
            },
            'customer_analysis': {
                'top_customers': top_customers,
                'country_analysis': country_analysis
            },
            'inventory_analysis': {
                'best_sellers': best_sellers,
                'stock_analysis': stock_analysis
            },
            'metrics': metrics
        }
    
    def create_visualizations(self, sales_by_category, monthly_trend, top_customers):
        """Create basic visualizations for the report"""
        try:
            # Set up the plot style
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('E-Commerce Analytics Dashboard', fontsize=16)
            
            # Plot 1: Sales by Category
            axes[0, 0].bar(sales_by_category['category'], sales_by_category['total_sales'])
            axes[0, 0].set_title('Sales by Category')
            axes[0, 0].set_ylabel('Total Sales ($)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Monthly Sales Trend
            axes[0, 1].plot(monthly_trend['month'], monthly_trend['monthly_sales'], marker='o')
            axes[0, 1].set_title('Monthly Sales Trend')
            axes[0, 1].set_ylabel('Monthly Sales ($)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Plot 3: Top Customers
            axes[1, 0].barh(top_customers['name'], top_customers['total_spent'])
            axes[1, 0].set_title('Top Customers by Spending')
            axes[1, 0].set_xlabel('Total Spent ($)')
            
            # Plot 4: Best Selling Products
            top_products = self.execute_query('''
                SELECT p.name, SUM(oi.quantity) as total_sold
                FROM products p
                JOIN order_items oi ON p.product_id = oi.product_id
                GROUP BY p.product_id
                ORDER BY total_sold DESC
                LIMIT 5
            ''')
            axes[1, 1].pie(top_products['total_sold'], labels=top_products['name'], autopct='%1.1f%%')
            axes[1, 1].set_title('Top 5 Products by Quantity Sold')
            
            plt.tight_layout()
            plt.savefig('ecommerce_analytics.png', dpi=300, bbox_inches='tight')
            print("\nVisualization saved as 'ecommerce_analytics.png'")
            plt.close()
            
        except Exception as e:
            print(f"Visualization creation failed: {e}")

def run_mini_project():
    """Run the complete e-commerce analytics mini project"""
    print("E-COMMERCE ANALYTICS MINI PROJECT")
    print("=" * 50)
    
    # Initialize the system
    ecommerce = ECommerceAnalytics()
    
    # Generate comprehensive report
    report = ecommerce.generate_report()
    
    # Demonstrate additional SQL operations
    print("\nADDITIONAL SQL OPERATIONS DEMONSTRATION")
    print("=" * 50)
    
    # Complex JOIN example
    print("\n1. Customer Order Details (JOIN example):")
    query = '''
        SELECT c.name, c.email, o.order_id, o.order_date, o.total_amount, o.status
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        ORDER BY o.order_date DESC
    '''
    order_details = ecommerce.execute_query(query)
    print(order_details.to_string(index=False))
    
    # Subquery example
    print("\n2. Products Never Ordered (Subquery example):")
    query = '''
        SELECT p.name, p.category, p.price
        FROM products p
        WHERE p.product_id NOT IN (
            SELECT DISTINCT product_id FROM order_items
        )
    '''
    never_ordered = ecommerce.execute_query(query)
    if len(never_ordered) > 0:
        print(never_ordered.to_string(index=False))
    else:
        print("All products have been ordered at least once!")
    
    # CASE statement example
    print("\n3. Customer Segmentation (CASE statement):")
    query = '''
        SELECT 
            c.name,
            c.country,
            SUM(o.total_amount) as total_spent,
            CASE 
                WHEN SUM(o.total_amount) > 1000 THEN 'VIP'
                WHEN SUM(o.total_amount) > 500 THEN 'Regular'
                ELSE 'New'
            END as customer_segment
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.customer_id
        ORDER BY total_spent DESC
    '''
    customer_segments = ecommerce.execute_query(query)
    print(customer_segments.to_string(index=False))
    
    print("\n" + "=" * 50)
    print("MINI PROJECT COMPLETED SUCCESSFULLY!")
    print("This project demonstrates end-to-end SQL operations including:")
    print("• Database creation and population")
    print("• Complex JOIN operations")
    print("• Aggregation and GROUP BY")
    print("• Subqueries and CASE statements")
    print("• Business analytics and reporting")
    print("• Data visualization")

if __name__ == "__main__":
    run_mini_project()