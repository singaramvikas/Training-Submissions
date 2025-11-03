"""
Hard 2: Build a mini project applying Python Data Structures end-to-end.
"""

import json
import csv
import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict, deque, Counter
import random
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib

class ProductCategory(Enum):
    ELECTRONICS = "Electronics"
    CLOTHING = "Clothing"
    BOOKS = "Books"
    HOME = "Home & Garden"
    SPORTS = "Sports & Outdoors"

@dataclass
class Product:
    product_id: str
    name: str
    category: ProductCategory
    price: float
    stock_quantity: int
    supplier: str

@dataclass
class Customer:
    customer_id: str
    name: str
    email: str
    join_date: datetime
    loyalty_points: int = 0

@dataclass
class Order:
    order_id: str
    customer_id: str
    order_date: datetime
    items: List[Tuple[str, int]]  # (product_id, quantity)
    total_amount: float
    status: str

class ECommerceSystem:
    """
    End-to-end e-commerce system demonstrating comprehensive use of Python data structures
    """
    
    def __init__(self):
        # Core data structures
        self.products: Dict[str, Product] = {}
        self.customers: Dict[str, Customer] = {}
        self.orders: Dict[str, Order] = {}
        
        # Indexing structures for fast lookups
        self.products_by_category: Dict[ProductCategory, Set[str]] = defaultdict(set)
        self.customer_orders: Dict[str, List[str]] = defaultdict(list)
        self.product_orders: Dict[str, List[str]] = defaultdict(list)
        
        # Analytics structures
        self.sales_history: deque = deque(maxlen=1000)  # Rolling window of sales
        self.inventory_changes: List[Tuple] = []  # Audit trail
        
        # Cache structures
        self._category_revenue_cache = {}
        self._customer_activity_cache = {}
        
        # Session management
        self.active_sessions: Dict[str, Dict] = {}  # session_id -> cart data
        
        # Initialize with sample data
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize the system with sample data"""
        print("Initializing e-commerce system with sample data...")
        
        # Sample products
        sample_products = [
            Product("P001", "Laptop", ProductCategory.ELECTRONICS, 999.99, 50, "TechSupplies Inc"),
            Product("P002", "T-Shirt", ProductCategory.CLOTHING, 19.99, 200, "FashionCo"),
            Product("P003", "Python Cookbook", ProductCategory.BOOKS, 49.99, 100, "BookHouse"),
            Product("P004", "Coffee Maker", ProductCategory.HOME, 79.99, 30, "HomeEssentials"),
            Product("P005", "Basketball", ProductCategory.SPORTS, 29.99, 75, "SportsGear"),
            Product("P006", "Smartphone", ProductCategory.ELECTRONICS, 699.99, 25, "TechSupplies Inc"),
            Product("P007", "Jeans", ProductCategory.CLOTHING, 59.99, 150, "FashionCo"),
            Product("P008", "Data Science Guide", ProductCategory.BOOKS, 39.99, 80, "BookHouse"),
        ]
        
        for product in sample_products:
            self.add_product(product)
        
        # Sample customers
        sample_customers = [
            Customer("C001", "Alice Johnson", "alice@email.com", datetime(2023, 1, 15)),
            Customer("C002", "Bob Smith", "bob@email.com", datetime(2023, 2, 20)),
            Customer("C003", "Carol Davis", "carol@email.com", datetime(2023, 3, 10)),
        ]
        
        for customer in sample_customers:
            self.add_customer(customer)
    
    def add_product(self, product: Product):
        """Add a product to the system"""
        self.products[product.product_id] = product
        self.products_by_category[product.category].add(product.product_id)
        print(f"Added product: {product.name}")
    
    def add_customer(self, customer: Customer):
        """Add a customer to the system"""
        self.customers[customer.customer_id] = customer
        print(f"Added customer: {customer.name}")
    
    def create_order(self, customer_id: str, items: List[Tuple[str, int]]) -> str:
        """Create a new order"""
        if customer_id not in self.customers:
            raise ValueError("Customer not found")
        
        # Calculate total and check stock
        total_amount = 0.0
        for product_id, quantity in items:
            if product_id not in self.products:
                raise ValueError(f"Product {product_id} not found")
            
            product = self.products[product_id]
            if product.stock_quantity < quantity:
                raise ValueError(f"Insufficient stock for {product.name}")
            
            total_amount += product.price * quantity
        
        # Generate order ID
        order_id = f"ORD{len(self.orders) + 1:06d}"
        
        # Create order
        order = Order(
            order_id=order_id,
            customer_id=customer_id,
            order_date=datetime.now(),
            items=items.copy(),
            total_amount=total_amount,
            status="Pending"
        )
        
        # Update data structures
        self.orders[order_id] = order
        self.customer_orders[customer_id].append(order_id)
        
        for product_id, quantity in items:
            self.product_orders[product_id].append(order_id)
        
        # Update inventory
        for product_id, quantity in items:
            self.products[product_id].stock_quantity -= quantity
            self.inventory_changes.append((
                product_id, quantity, "sale", datetime.now()
            ))
        
        # Add to sales history
        self.sales_history.append((order_id, total_amount, datetime.now()))
        
        # Invalidate caches
        self._category_revenue_cache.clear()
        self._customer_activity_cache.clear()
        
        print(f"Created order {order_id} for customer {customer_id}: ${total_amount:.2f}")
        return order_id
    
    def get_category_revenue(self) -> Dict[ProductCategory, float]:
        """Calculate revenue by category (with caching)"""
        if self._category_revenue_cache:
            return self._category_revenue_cache
        
        revenue_by_category = defaultdict(float)
        
        for order in self.orders.values():
            for product_id, quantity in order.items:
                product = self.products[product_id]
                revenue_by_category[product.category] += product.price * quantity
        
        self._category_revenue_cache = dict(revenue_by_category)
        return self._category_revenue_cache
    
    def get_customer_activity(self, customer_id: str) -> Dict:
        """Get customer activity summary (with caching)"""
        cache_key = customer_id
        if cache_key in self._customer_activity_cache:
            return self._customer_activity_cache[cache_key]
        
        if customer_id not in self.customers:
            raise ValueError("Customer not found")
        
        customer_orders = [self.orders[oid] for oid in self.customer_orders[customer_id]]
        
        activity = {
            'total_orders': len(customer_orders),
            'total_spent': sum(order.total_amount for order in customer_orders),
            'average_order_value': 0,
            'favorite_category': None,
            'order_frequency': None
        }
        
        if customer_orders:
            activity['average_order_value'] = activity['total_spent'] / activity['total_orders']
            
            # Find favorite category
            category_counter = Counter()
            for order in customer_orders:
                for product_id, quantity in order.items:
                    product = self.products[product_id]
                    category_counter[product.category] += quantity
            
            if category_counter:
                activity['favorite_category'] = category_counter.most_common(1)[0][0]
        
        self._customer_activity_cache[cache_key] = activity
        return activity
    
    def generate_sales_report(self, days: int = 30) -> Dict:
        """Generate comprehensive sales report"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_orders = [
            order for order in self.orders.values() 
            if order.order_date >= cutoff_date
        ]
        
        # Calculate various metrics
        total_revenue = sum(order.total_amount for order in recent_orders)
        total_orders = len(recent_orders)
        
        # Product performance
        product_sales = Counter()
        for order in recent_orders:
            for product_id, quantity in order.items:
                product_sales[product_id] += quantity
        
        top_products = [
            (self.products[pid].name, quantity) 
            for pid, quantity in product_sales.most_common(5)
        ]
        
        # Customer analytics
        customer_orders = Counter()
        for order in recent_orders:
            customer_orders[order.customer_id] += 1
        
        top_customers = [
            (self.customers[cid].name, count) 
            for cid, count in customer_orders.most_common(5)
        ]
        
        report = {
            'period_days': days,
            'total_revenue': total_revenue,
            'total_orders': total_orders,
            'average_order_value': total_revenue / total_orders if total_orders else 0,
            'top_products': top_products,
            'top_customers': top_customers,
            'category_revenue': self.get_category_revenue(),
            'generated_at': datetime.now()
        }
        
        return report
    
    def recommend_products(self, customer_id: str, max_recommendations: int = 5) -> List[Product]:
        """Simple recommendation system based on purchase history"""
        if customer_id not in self.customer_orders:
            return []
        
        # Get customer's purchased categories
        customer_categories = set()
        for order_id in self.customer_orders[customer_id]:
            order = self.orders[order_id]
            for product_id, _ in order.items:
                product = self.products[product_id]
                customer_categories.add(product.category)
        
        # Find products in those categories that customer hasn't purchased
        recommended_products = []
        seen_products = set()
        
        # Get all products customer has ordered
        for order_id in self.customer_orders[customer_id]:
            order = self.orders[order_id]
            for product_id, _ in order.items:
                seen_products.add(product_id)
        
        # Recommend products from favorite categories
        for category in customer_categories:
            for product_id in self.products_by_category[category]:
                if (product_id not in seen_products and 
                    self.products[product_id].stock_quantity > 0):
                    recommended_products.append(self.products[product_id])
        
        # If not enough recommendations, add popular products
        if len(recommended_products) < max_recommendations:
            all_products = list(self.products.values())
            for product in all_products:
                if (product.product_id not in seen_products and 
                    product.stock_quantity > 0 and 
                    product not in recommended_products):
                    recommended_products.append(product)
        
        return recommended_products[:max_recommendations]
    
    def export_data(self, format_type: str = "json"):
        """Export system data in specified format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == "json":
            filename = f"ecommerce_export_{timestamp}.json"
            export_data = {
                'products': {
                    pid: {
                        'name': p.name,
                        'category': p.category.value,
                        'price': p.price,
                        'stock': p.stock_quantity
                    } for pid, p in self.products.items()
                },
                'customers': {
                    cid: {
                        'name': c.name,
                        'email': c.email,
                        'join_date': c.join_date.isoformat()
                    } for cid, c in self.customers.items()
                },
                'analytics': {
                    'total_orders': len(self.orders),
                    'total_revenue': sum(o.total_amount for o in self.orders.values()),
                    'category_revenue': self.get_category_revenue()
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"Data exported to {filename}")
        
        elif format_type == "csv":
            # Export products
            products_filename = f"products_export_{timestamp}.csv"
            with open(products_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['ProductID', 'Name', 'Category', 'Price', 'Stock'])
                for product in self.products.values():
                    writer.writerow([
                        product.product_id,
                        product.name,
                        product.category.value,
                        product.price,
                        product.stock_quantity
                    ])
            
            print(f"Products exported to {products_filename}")
    
    def run_demo(self):
        """Run a comprehensive demonstration of the system"""
        print("\n" + "=" * 60)
        print("E-COMMERCE SYSTEM DEMONSTRATION")
        print("=" * 60)
        
        # 1. Show initial state
        print(f"\n1. INITIAL SYSTEM STATE:")
        print(f"   Products: {len(self.products)}")
        print(f"   Customers: {len(self.customers)}")
        print(f"   Orders: {len(self.orders)}")
        
        # 2. Simulate some orders
        print(f"\n2. SIMULATING ORDERS:")
        orders_to_create = [
            ("C001", [("P001", 1), ("P003", 2)]),  # Alice buys laptop and books
            ("C002", [("P002", 3), ("P007", 1)]),  # Bob buys clothing
            ("C001", [("P006", 1)]),  # Alice buys smartphone
            ("C003", [("P004", 1), ("P005", 2)]),  # Carol buys home and sports
        ]
        
        for customer_id, items in orders_to_create:
            try:
                order_id = self.create_order(customer_id, items)
                print(f"   Created {order_id}")
            except ValueError as e:
                print(f"   Failed to create order: {e}")
        
        # 3. Show analytics
        print(f"\n3. SYSTEM ANALYTICS:")
        report = self.generate_sales_report(days=365)
        print(f"   Total Revenue: ${report['total_revenue']:.2f}")
        print(f"   Total Orders: {report['total_orders']}")
        print(f"   Average Order Value: ${report['average_order_value']:.2f}")
        
        print(f"\n   Category Revenue:")
        for category, revenue in report['category_revenue'].items():
            print(f"     {category.value}: ${revenue:.2f}")
        
        print(f"\n   Top Products:")
        for product, quantity in report['top_products']:
            print(f"     {product}: {quantity} units")
        
        # 4. Demonstrate recommendations
        print(f"\n4. PRODUCT RECOMMENDATIONS:")
        for customer_id in self.customers:
            recommendations = self.recommend_products(customer_id, 3)
            if recommendations:
                customer_name = self.customers[customer_id].name
                print(f"   For {customer_name}:")
                for product in recommendations:
                    print(f"     - {product.name} (${product.price:.2f})")
        
        # 5. Customer activity
        print(f"\n5. CUSTOMER ACTIVITY:")
        for customer_id in self.customers:
            activity = self.get_customer_activity(customer_id)
            customer = self.customers[customer_id]
            print(f"   {customer.name}:")
            print(f"     Orders: {activity['total_orders']}")
            print(f"     Total Spent: ${activity['total_spent']:.2f}")
            if activity['favorite_category']:
                print(f"     Favorite Category: {activity['favorite_category'].value}")
        
        # 6. Export data
        print(f"\n6. DATA EXPORT:")
        self.export_data("json")
        self.export_data("csv")
        
        print(f"\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)

def main():
    """Main execution function"""
    ecommerce_system = ECommerceSystem()
    ecommerce_system.run_demo()

if __name__ == "__main__":
    main()