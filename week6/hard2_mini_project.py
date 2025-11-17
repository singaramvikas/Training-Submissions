import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ECommerceAnalyticsPlatform:
    """End-to-end e-commerce analytics platform using Advanced SQL Joins & Aggregations"""
    
    def __init__(self, db_path='ecommerce_analytics.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._setup_database()
    
    def _setup_database(self):
        """Initialize database schema"""
        # Enable WAL mode for better concurrent performance
        self.conn.execute('PRAGMA journal_mode=WAL;')
        self.conn.execute('PRAGMA synchronous=NORMAL;')
        
    def generate_sample_data(self, days=365, n_customers=5000, n_products=200):
        """Generate comprehensive e-commerce sample data"""
        print("Generating sample e-commerce data...")
        
        np.random.seed(42)  # For reproducible results
        
        # Products dimension table
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports', 'Beauty']
        products_data = []
        
        for i in range(1, n_products + 1):
            category = np.random.choice(categories)
            base_price = np.random.lognormal(3, 1)
            products_data.append({
                'product_id': i,
                'product_name': f'Product_{i:04d}',
                'category': category,
                'subcategory': f'{category}_Sub_{np.random.randint(1, 6)}',
                'price': round(base_price, 2),
                'cost': round(base_price * np.random.uniform(0.3, 0.7), 2),
                'supplier_id': np.random.randint(1, 51)
            })
        
        products_df = pd.DataFrame(products_data)
        products_df.to_sql('products', self.conn, index=False, if_exists='replace')
        
        # Customers dimension table
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia']
        segments = ['Premium', 'Standard', 'Basic']
        
        customers_data = []
        for i in range(1, n_customers + 1):
            signup_date = datetime(2023, 1, 1) - timedelta(days=np.random.randint(0, 365))
            customers_data.append({
                'customer_id': i,
                'name': f'Customer_{i:05d}',
                'email': f'customer_{i:05d}@example.com',
                'city': np.random.choice(cities),
                'segment': np.random.choice(segments, p=[0.2, 0.5, 0.3]),
                'signup_date': signup_date.date(),
                'loyalty_tier': np.random.choice(['Gold', 'Silver', 'Bronze'], p=[0.1, 0.3, 0.6])
            })
        
        customers_df = pd.DataFrame(customers_data)
        customers_df.to_sql('customers', self.conn, index=False, if_exists='replace')
        
        # Generate orders fact table
        orders_data = []
        order_items_data = []
        order_id = 1
        start_date = datetime(2023, 1, 1)
        
        for day in range(days):
            daily_orders = np.random.poisson(50)  # Average 50 orders per day
            
            for _ in range(daily_orders):
                customer_id = np.random.randint(1, n_customers + 1)
                order_date = start_date + timedelta(days=day, hours=np.random.randint(0, 24))
                status = np.random.choice(['completed', 'pending', 'cancelled'], p=[0.85, 0.1, 0.05])
                
                orders_data.append({
                    'order_id': order_id,
                    'customer_id': customer_id,
                    'order_date': order_date,
                    'status': status,
                    'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Bank Transfer'])
                })
                
                # Generate order items
                items_in_order = np.random.randint(1, 6)
                for item_num in range(items_in_order):
                    product_id = np.random.randint(1, n_products + 1)
                    quantity = np.random.randint(1, 4)
                    unit_price = products_df[products_df['product_id'] == product_id]['price'].values[0]
                    
                    order_items_data.append({
                        'order_item_id': len(order_items_data) + 1,
                        'order_id': order_id,
                        'product_id': product_id,
                        'quantity': quantity,
                        'unit_price': unit_price,
                        'discount': round(np.random.uniform(0, 0.3), 2)
                    })
                
                order_id += 1
        
        orders_df = pd.DataFrame(orders_data)
        orders_df.to_sql('orders', self.conn, index=False, if_exists='replace')
        
        order_items_df = pd.DataFrame(order_items_data)
        order_items_df.to_sql('order_items', self.conn, index=False, if_exists='replace')
        
        # Create indexes for performance
        self._create_indexes()
        
        print(f"Data generation complete:")
        print(f"  - Products: {len(products_df):,}")
        print(f"  - Customers: {len(customers_df):,}")
        print(f"  - Orders: {len(orders_df):,}")
        print(f"  - Order items: {len(order_items_df):,}")
        
        return products_df, customers_df, orders_df, order_items_df
    
    def _create_indexes(self):
        """Create optimal indexes for query performance"""
        indexes = [
            "CREATE INDEX idx_orders_customer_id ON orders(customer_id)",
            "CREATE INDEX idx_orders_date ON orders(order_date)",
            "CREATE INDEX idx_orders_status ON orders(status)",
            "CREATE INDEX idx_order_items_order_id ON order_items(order_id)",
            "CREATE INDEX idx_order_items_product_id ON order_items(product_id)",
            "CREATE INDEX idx_customers_city ON customers(city)",
            "CREATE INDEX idx_customers_segment ON customers(segment)",
            "CREATE INDEX idx_products_category ON products(category)"
        ]
        
        for index_sql in indexes:
            try:
                self.conn.execute(index_sql)
            except:
                pass  # Index might already exist
    
    def advanced_sales_analytics(self):
        """Perform comprehensive sales analytics using advanced SQL"""
        analytics_queries = {
            'customer_lifetime_value': """
                WITH customer_metrics AS (
                    SELECT 
                        c.customer_id,
                        c.name,
                        c.segment,
                        c.city,
                        COUNT(DISTINCT o.order_id) as total_orders,
                        SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) as total_spent,
                        MIN(o.order_date) as first_order_date,
                        MAX(o.order_date) as last_order_date,
                        AVG(oi.quantity * oi.unit_price * (1 - oi.discount)) as avg_order_value
                    FROM customers c
                    JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
                    JOIN order_items oi ON o.order_id = oi.order_id
                    GROUP BY c.customer_id, c.name, c.segment, c.city
                )
                SELECT 
                    segment,
                    city,
                    COUNT(*) as customer_count,
                    AVG(total_orders) as avg_orders_per_customer,
                    AVG(total_spent) as avg_lifetime_value,
                    AVG(total_spent / NULLIF(total_orders, 0)) as avg_order_value,
                    SUM(total_spent) as total_segment_revenue
                FROM customer_metrics
                GROUP BY segment, city
                ORDER BY total_segment_revenue DESC
            """,
            
            'product_performance_analysis': """
                WITH product_sales AS (
                    SELECT 
                        p.product_id,
                        p.product_name,
                        p.category,
                        p.subcategory,
                        COUNT(DISTINCT oi.order_id) as times_ordered,
                        SUM(oi.quantity) as total_quantity,
                        SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) as total_revenue,
                        SUM(oi.quantity * (oi.unit_price - p.cost)) as total_profit,
                        AVG(oi.quantity * oi.unit_price * (1 - oi.discount)) as avg_sale_amount
                    FROM products p
                    JOIN order_items oi ON p.product_id = oi.product_id
                    JOIN orders o ON oi.order_id = o.order_id AND o.status = 'completed'
                    GROUP BY p.product_id, p.product_name, p.category, p.subcategory
                )
                SELECT 
                    category,
                    subcategory,
                    COUNT(*) as product_count,
                    SUM(total_quantity) as total_units_sold,
                    SUM(total_revenue) as category_revenue,
                    SUM(total_profit) as category_profit,
                    RANK() OVER (ORDER BY SUM(total_revenue) DESC) as revenue_rank
                FROM product_sales
                GROUP BY category, subcategory
                ORDER BY category_revenue DESC
            """,
            
            'monthly_sales_trends': """
                SELECT 
                    strftime('%Y-%m', o.order_date) as sales_month,
                    p.category,
                    COUNT(DISTINCT o.order_id) as order_count,
                    COUNT(DISTINCT o.customer_id) as unique_customers,
                    SUM(oi.quantity) as total_units,
                    SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) as monthly_revenue,
                    SUM(oi.quantity * (oi.unit_price - p.cost)) as monthly_profit,
                    LAG(SUM(oi.quantity * oi.unit_price * (1 - oi.discount))) 
                        OVER (PARTITION BY p.category ORDER BY strftime('%Y-%m', o.order_date)) as prev_month_revenue
                FROM orders o
                JOIN order_items oi ON o.order_id = oi.order_id
                JOIN products p ON oi.product_id = p.product_id
                WHERE o.status = 'completed'
                GROUP BY sales_month, p.category
                ORDER BY sales_month, p.category
            """,
            
            'customer_retention_analysis': """
                WITH customer_monthly_activity AS (
                    SELECT 
                        customer_id,
                        strftime('%Y-%m', order_date) as activity_month,
                        COUNT(DISTINCT order_id) as monthly_orders
                    FROM orders 
                    WHERE status = 'completed'
                    GROUP BY customer_id, activity_month
                ),
                monthly_cohorts AS (
                    SELECT 
                        customer_id,
                        MIN(activity_month) as cohort_month
                    FROM customer_monthly_activity
                    GROUP BY customer_id
                )
                SELECT 
                    mc.cohort_month,
                    cma.activity_month,
                    COUNT(DISTINCT cma.customer_id) as active_customers,
                    ROUND(100.0 * COUNT(DISTINCT cma.customer_id) / 
                          FIRST_VALUE(COUNT(DISTINCT cma.customer_id)) 
                          OVER (PARTITION BY mc.cohort_month ORDER BY cma.activity_month), 2) as retention_rate
                FROM monthly_cohorts mc
                JOIN customer_monthly_activity cma ON mc.customer_id = cma.customer_id
                GROUP BY mc.cohort_month, cma.activity_month
                ORDER BY mc.cohort_month, cma.activity_month
            """
        }
        
        analytics_results = {}
        for analysis_name, query in analytics_queries.items():
            print(f"Running {analysis_name}...")
            analytics_results[analysis_name] = pd.read_sql_query(query, self.conn)
        
        return analytics_results
    
    def create_dashboard_visualizations(self, analytics_results):
        """Create comprehensive visualizations for the analytics dashboard"""
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Customer Lifetime Value by Segment
        plt.subplot(3, 3, 1)
        clv_data = analytics_results['customer_lifetime_value']
        segments_pivot = clv_data.pivot_table(
            values='avg_lifetime_value', 
            index='city', 
            columns='segment', 
            aggfunc='mean'
        ).fillna(0)
        segments_pivot.plot(kind='bar', ax=plt.gca())
        plt.title('Average Customer Lifetime Value\nby Segment and City')
        plt.xticks(rotation=45)
        plt.ylabel('Average CLV ($)')
        plt.legend(title='Segment')
        
        # 2. Product Category Performance
        plt.subplot(3, 3, 2)
        product_data = analytics_results['product_performance_analysis']
        categories = product_data.groupby('category')[['category_revenue', 'category_profit']].sum()
        categories.plot(kind='bar', ax=plt.gca())
        plt.title('Revenue vs Profit by Category')
        plt.xticks(rotation=45)
        plt.ylabel('Amount ($)')
        plt.legend(['Revenue', 'Profit'])
        
        # 3. Monthly Sales Trends
        plt.subplot(3, 3, 3)
        monthly_data = analytics_results['monthly_sales_trends']
        monthly_pivot = monthly_data.pivot(index='sales_month', columns='category', values='monthly_revenue')
        monthly_pivot.plot(kind='line', ax=plt.gca(), marker='o')
        plt.title('Monthly Revenue Trends by Category')
        plt.xticks(rotation=45)
        plt.ylabel('Revenue ($)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Customer Retention Heatmap
        plt.subplot(3, 3, 4)
        retention_data = analytics_results['customer_retention_analysis']
        retention_pivot = retention_data.pivot_table(
            values='retention_rate', 
            index='cohort_month', 
            columns='activity_month', 
            aggfunc='mean'
        ).fillna(0)
        sns.heatmap(retention_pivot, annot=True, fmt='.1f', cmap='YlGnBu', ax=plt.gca())
        plt.title('Customer Retention Rate Heatmap (%)')
        plt.xlabel('Activity Month')
        plt.ylabel('Cohort Month')
        
        # 5. Revenue Composition
        plt.subplot(3, 3, 5)
        category_revenue = product_data.groupby('category')['category_revenue'].sum()
        plt.pie(category_revenue.values, labels=category_revenue.index, autopct='%1.1f%%')
        plt.title('Revenue Distribution by Category')
        
        # 6. Monthly Growth Rates
        plt.subplot(3, 3, 6)
        monthly_data['revenue_growth'] = (
            (monthly_data['monthly_revenue'] - monthly_data['prev_month_revenue']) 
            / monthly_data['prev_month_revenue'] * 100
        )
        growth_pivot = monthly_data.pivot(index='sales_month', columns='category', values='revenue_growth')
        growth_pivot.plot(kind='line', ax=plt.gca(), marker='s')
        plt.title('Monthly Revenue Growth Rate (%)')
        plt.xticks(rotation=45)
        plt.ylabel('Growth Rate (%)')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 7. Customer Segment Distribution
        plt.subplot(3, 3, 7)
        segment_dist = clv_data.groupby('segment')['customer_count'].sum()
        plt.bar(segment_dist.index, segment_dist.values)
        plt.title('Customer Distribution by Segment')
        plt.ylabel('Number of Customers')
        
        # 8. Profit Margins by Category
        plt.subplot(3, 3, 8)
        product_data['profit_margin'] = (product_data['category_profit'] / product_data['category_revenue']) * 100
        margin_by_category = product_data.groupby('category')['profit_margin'].mean()
        plt.bar(margin_by_category.index, margin_by_category.values)
        plt.title('Average Profit Margin by Category')
        plt.ylabel('Profit Margin (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('ecommerce_analytics_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_business_insights(self, analytics_results):
        """Generate actionable business insights from analytics"""
        print("\n" + "="*60)
        print("BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        clv_data = analytics_results['customer_lifetime_value']
        product_data = analytics_results['product_performance_analysis']
        monthly_data = analytics_results['monthly_sales_trends']
        retention_data = analytics_results['customer_retention_analysis']
        
        # Customer Insights
        print("\n1. CUSTOMER INSIGHTS:")
        top_segment = clv_data.loc[clv_data['total_segment_revenue'].idxmax()]
        print(f"   • Highest revenue segment: {top_segment['segment']} in {top_segment['city']}")
        print(f"     (Revenue: ${top_segment['total_segment_revenue']:,.2f})")
        
        avg_clv = clv_data['avg_lifetime_value'].mean()
        print(f"   • Average Customer Lifetime Value: ${avg_clv:,.2f}")
        
        # Product Insights
        print("\n2. PRODUCT PERFORMANCE:")
        top_category = product_data.loc[product_data['category_revenue'].idxmax()]
        print(f"   • Top performing category: {top_category['category']}")
        print(f"     (Revenue: ${top_category['category_revenue']:,.2f})")
        
        high_margin_categories = product_data[product_data['profit_margin'] > 30]
        if not high_margin_categories.empty:
            print(f"   • High-margin categories (>30%): {', '.join(high_margin_categories['category'].unique())}")
        
        # Sales Trends
        print("\n3. SALES TRENDS:")
        recent_month = monthly_data['sales_month'].max()
        recent_growth = monthly_data[monthly_data['sales_month'] == recent_month]['revenue_growth'].mean()
        print(f"   • Most recent month ({recent_month}): {recent_growth:.1f}% revenue growth")
        
        # Retention Insights
        print("\n4. CUSTOMER RETENTION:")
        avg_retention = retention_data['retention_rate'].mean()
        print(f"   • Average customer retention rate: {avg_retention:.1f}%")
        
        # Recommendations
        print("\n5. STRATEGIC RECOMMENDATIONS:")
        if top_segment['segment'] == 'Premium':
            print("   • Focus on premium customer acquisition and retention")
        if high_margin_categories.empty:
            print("   • Investigate opportunities to improve profit margins")
        if avg_retention < 50:
            print("   • Implement customer retention programs")
        
        return {
            'top_segment': top_segment,
            'top_category': top_category,
            'recent_growth': recent_growth,
            'avg_retention': avg_retention
        }
    
    def run_complete_analysis(self):
        """Run the complete end-to-end analysis"""
        print("=== E-COMMERCE ANALYTICS PLATFORM ===")
        print("Advanced SQL Joins & Aggregations - End-to-End Project\n")
        
        # Step 1: Generate sample data
        self.generate_sample_data(days=180, n_customers=2000, n_products=100)
        
        # Step 2: Perform advanced analytics
        analytics_results = self.advanced_sales_analytics()
        
        # Step 3: Create visualizations
        self.create_dashboard_visualizations(analytics_results)
        
        # Step 4: Generate business insights
        insights = self.generate_business_insights(analytics_results)
        
        print(f"\nAnalysis complete! Check 'ecommerce_analytics_dashboard.png' for visualizations.")
        
        return analytics_results, insights

# Execute the complete project
if __name__ == "__main__":
    # Initialize the analytics platform
    platform = ECommerceAnalyticsPlatform()
    
    # Run complete analysis
    results, insights = platform.run_complete_analysis()
    
    # Close database connection
    platform.conn.close()