"""
Hard 2: Build a mini project applying Pandas Fundamentals end-to-end
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class ECommerceAnalytics:
    """
    End-to-end e-commerce analytics mini-project demonstrating Pandas fundamentals
    """
    
    def __init__(self):
        self.df = None
        self.cleaned_df = None
        self.analysis_results = {}
        
    def generate_sample_data(self, n_customers=10000, n_products=500):
        """
        Generate realistic e-commerce sample data
        """
        print("Generating sample e-commerce data...")
        np.random.seed(42)
        
        # Customer data
        customers = []
        for i in range(n_customers):
            customers.append({
                'customer_id': f'CUST_{i:05d}',
                'age': np.random.randint(18, 70),
                'gender': np.random.choice(['M', 'F'], p=[0.48, 0.52]),
                'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], 
                                       p=[0.2, 0.15, 0.15, 0.3, 0.2]),
                'signup_date': pd.to_datetime('2020-01-01') + pd.to_timedelta(np.random.randint(0, 1095), unit='d'),
                'loyalty_tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], 
                                               p=[0.4, 0.3, 0.2, 0.1])
            })
        
        customer_df = pd.DataFrame(customers)
        
        # Product data
        products = []
        categories = ['Electronics', 'Clothing', 'Home', 'Books', 'Sports']
        subcategories = {
            'Electronics': ['Phones', 'Laptops', 'Accessories'],
            'Clothing': ['Men', 'Women', 'Kids'],
            'Home': ['Furniture', 'Kitchen', 'Decor'],
            'Books': ['Fiction', 'Non-Fiction', 'Educational'],
            'Sports': ['Outdoor', 'Fitness', 'Team Sports']
        }
        
        for i in range(n_products):
            category = np.random.choice(categories)
            subcategory = np.random.choice(subcategories[category])
            products.append({
                'product_id': f'PROD_{i:05d}',
                'product_name': f'{subcategory} Product {i}',
                'category': category,
                'subcategory': subcategory,
                'price': np.random.lognormal(3, 1).clip(5, 1000),
                'cost': 0,  # Will be calculated based on price
            })
        
        product_df = pd.DataFrame(products)
        product_df['cost'] = product_df['price'] * np.random.uniform(0.3, 0.7)
        
        # Transaction data
        transactions = []
        n_transactions = n_customers * 12  # Average 12 transactions per customer
        
        for i in range(n_transactions):
            customer = np.random.choice(customer_df['customer_id'])
            product = np.random.choice(product_df['product_id'])
            product_data = product_df[product_df['product_id'] == product].iloc[0]
            
            transaction_date = pd.to_datetime('2022-01-01') + pd.to_timedelta(np.random.randint(0, 730), unit='d')
            
            transactions.append({
                'transaction_id': f'TRX_{i:06d}',
                'customer_id': customer,
                'product_id': product,
                'transaction_date': transaction_date,
                'quantity': np.random.poisson(1.5).clip(1, 10),
                'discount_pct': np.random.choice([0, 5, 10, 15, 20], p=[0.5, 0.2, 0.15, 0.1, 0.05]),
                'returned': np.random.choice([True, False], p=[0.05, 0.95])
            })
        
        transaction_df = pd.DataFrame(transactions)
        
        # Merge all data
        self.df = transaction_df.merge(customer_df, on='customer_id').merge(product_df, on='product_id')
        
        # Calculate derived columns
        self.df['revenue'] = self.df['price'] * self.df['quantity'] * (1 - self.df['discount_pct'] / 100)
        self.df['profit'] = (self.df['price'] - self.df['cost']) * self.df['quantity'] * (1 - self.df['discount_pct'] / 100)
        self.df['month'] = self.df['transaction_date'].dt.to_period('M')
        self.df['day_of_week'] = self.df['transaction_date'].dt.day_name()
        
        print(f"Generated dataset with {len(self.df):,} transactions")
        return self.df
    
    def data_quality_check(self):
        """
        Comprehensive data quality assessment
        """
        print("\n" + "="*50)
        print("DATA QUALITY ASSESSMENT")
        print("="*50)
        
        quality_report = {}
        
        # Basic info
        quality_report['shape'] = self.df.shape
        quality_report['columns'] = list(self.df.columns)
        
        # Missing values
        missing_data = self.df.isnull().sum()
        quality_report['missing_values'] = missing_data[missing_data > 0].to_dict()
        
        # Data types
        quality_report['dtypes'] = self.df.dtypes.to_dict()
        
        # Duplicates
        quality_report['duplicate_rows'] = self.df.duplicated().sum()
        
        # Statistical summary
        quality_report['numeric_summary'] = self.df.describe()
        quality_report['categorical_summary'] = self.df.select_dtypes(include=['object']).describe()
        
        # Data quality issues
        issues = []
        if len(quality_report['missing_values']) > 0:
            issues.append(f"Missing values in {list(quality_report['missing_values'].keys())}")
        if quality_report['duplicate_rows'] > 0:
            issues.append(f"{quality_report['duplicate_rows']} duplicate rows found")
        
        # Check for negative revenue/profit
        negative_revenue = (self.df['revenue'] < 0).sum()
        if negative_revenue > 0:
            issues.append(f"{negative_revenue} transactions with negative revenue")
        
        quality_report['issues'] = issues
        
        # Print report
        print(f"Dataset Shape: {quality_report['shape']}")
        print(f"Missing Values: {quality_report['missing_values']}")
        print(f"Duplicate Rows: {quality_report['duplicate_rows']}")
        print(f"Data Quality Issues: {issues if issues else 'None'}")
        
        self.analysis_results['quality_report'] = quality_report
        return quality_report
    
    def clean_data(self):
        """
        Clean and preprocess the data
        """
        print("\n" + "="*50)
        print("DATA CLEANING AND PREPROCESSING")
        print("="*50)
        
        self.cleaned_df = self.df.copy()
        
        # Handle duplicates
        initial_size = len(self.cleaned_df)
        self.cleaned_df = self.cleaned_df.drop_duplicates()
        print(f"Removed {initial_size - len(self.cleaned_df)} duplicate rows")
        
        # Handle missing values (if any existed)
        for col in self.cleaned_df.columns:
            if self.cleaned_df[col].isnull().sum() > 0:
                if self.cleaned_df[col].dtype in ['float64', 'int64']:
                    self.cleaned_df[col].fillna(self.cleaned_df[col].median(), inplace=True)
                else:
                    self.cleaned_df[col].fillna(self.cleaned_df[col].mode()[0], inplace=True)
        
        # Remove outliers (transactions with negative revenue)
        outlier_count = (self.cleaned_df['revenue'] < 0).sum()
        self.cleaned_df = self.cleaned_df[self.cleaned_df['revenue'] >= 0]
        print(f"Removed {outlier_count} transactions with negative revenue")
        
        # Feature engineering
        self.cleaned_df['transaction_value'] = self.cleaned_df['revenue']
        self.cleaned_df['is_weekend'] = self.cleaned_df['day_of_week'].isin(['Saturday', 'Sunday'])
        self.cleaned_df['discount_level'] = pd.cut(self.cleaned_df['discount_pct'], 
                                                 bins=[-1, 0, 10, 20, 101], 
                                                 labels=['No Discount', 'Low', 'Medium', 'High'])
        
        print(f"Final cleaned dataset: {len(self.cleaned_df):,} rows")
        return self.cleaned_df
    
    def exploratory_analysis(self):
        """
        Comprehensive exploratory data analysis
        """
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        analysis = {}
        
        # 1. Overall business metrics
        analysis['total_revenue'] = self.cleaned_df['revenue'].sum()
        analysis['total_profit'] = self.cleaned_df['profit'].sum()
        analysis['total_transactions'] = len(self.cleaned_df)
        analysis['avg_transaction_value'] = self.cleaned_df['revenue'].mean()
        analysis['return_rate'] = self.cleaned_df['returned'].mean()
        
        print("1. Business Overview:")
        print(f"   Total Revenue: ${analysis['total_revenue']:,.2f}")
        print(f"   Total Profit: ${analysis['total_profit']:,.2f}")
        print(f"   Total Transactions: {analysis['total_transactions']:,}")
        print(f"   Average Transaction Value: ${analysis['avg_transaction_value']:.2f}")
        print(f"   Return Rate: {analysis['return_rate']:.2%}")
        
        # 2. Time series analysis
        monthly_metrics = self.cleaned_df.groupby('month').agg({
            'revenue': 'sum',
            'profit': 'sum',
            'transaction_id': 'count'
        }).rename(columns={'transaction_id': 'transaction_count'})
        
        analysis['monthly_metrics'] = monthly_metrics
        
        print(f"\n2. Monthly Performance (last 5 months):")
        print(monthly_metrics.tail())
        
        # 3. Customer analysis
        customer_metrics = self.cleaned_df.groupby('customer_id').agg({
            'revenue': 'sum',
            'transaction_id': 'count',
            'profit': 'sum'
        }).rename(columns={
            'transaction_id': 'transaction_count',
            'revenue': 'total_spent'
        })
        
        analysis['avg_customer_value'] = customer_metrics['total_spent'].mean()
        analysis['repeat_customers'] = (customer_metrics['transaction_count'] > 1).sum()
        
        print(f"\n3. Customer Insights:")
        print(f"   Average Customer Lifetime Value: ${analysis['avg_customer_value']:.2f}")
        print(f"   Repeat Customers: {analysis['repeat_customers']:,}")
        
        # 4. Product analysis
        category_performance = self.cleaned_df.groupby('category').agg({
            'revenue': 'sum',
            'profit': 'sum',
            'transaction_id': 'count'
        }).sort_values('revenue', ascending=False)
        
        analysis['category_performance'] = category_performance
        
        print(f"\n4. Top Performing Categories:")
        print(category_performance.head())
        
        # 5. Discount impact analysis
        discount_impact = self.cleaned_df.groupby('discount_level').agg({
            'revenue': 'sum',
            'transaction_id': 'count',
            'quantity': 'mean'
        })
        
        analysis['discount_impact'] = discount_impact
        
        self.analysis_results['exploratory_analysis'] = analysis
        return analysis
    
    def build_customer_segmentation(self):
        """
        Build customer segmentation using RFM analysis
        """
        print("\n" + "="*50)
        print("CUSTOMER SEGMENTATION (RFM ANALYSIS)")
        print("="*50)
        
        # Calculate RFM metrics
        max_date = self.cleaned_df['transaction_date'].max()
        
        rfm = self.cleaned_df.groupby('customer_id').agg({
            'transaction_date': lambda x: (max_date - x.max()).days,  # Recency
            'transaction_id': 'count',  # Frequency
            'revenue': 'sum'  # Monetary
        }).rename(columns={
            'transaction_date': 'recency',
            'transaction_id': 'frequency',
            'revenue': 'monetary'
        })
        
        # Create RFM scores
        rfm['r_score'] = pd.qcut(rfm['recency'], 4, labels=[4, 3, 2, 1])
        rfm['f_score'] = pd.qcut(rfm['frequency'], 4, labels=[1, 2, 3, 4])
        rfm['m_score'] = pd.qcut(rfm['monetary'], 4, labels=[1, 2, 3, 4])
        
        rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
        
        # Segment customers
        def segment_customer(row):
            if row['r_score'] == 4 and row['f_score'] in [3, 4] and row['m_score'] in [3, 4]:
                return 'Champions'
            elif row['r_score'] in [3, 4] and row['f_score'] in [2, 3, 4]:
                return 'Loyal Customers'
            elif row['r_score'] == 4:
                return 'New Customers'
            elif row['r_score'] in [2, 3] and row['f_score'] in [1, 2]:
                return 'At Risk'
            elif row['r_score'] in [1, 2] and row['f_score'] in [1, 2]:
                return 'Lost Customers'
            else:
                return 'Potential Loyalists'
        
        rfm['segment'] = rfm.apply(segment_customer, axis=1)
        
        segment_summary = rfm.groupby('segment').agg({
            'customer_id': 'count',
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean'
        }).rename(columns={'customer_id': 'customer_count'})
        
        print("Customer Segments:")
        print(segment_summary.round(2))
        
        self.analysis_results['customer_segmentation'] = rfm
        return rfm
    
    def predictive_analytics(self):
        """
        Build predictive models for customer behavior
        """
        print("\n" + "="*50)
        print("PREDICTIVE ANALYTICS")
        print("="*50)
        
        # Prepare data for modeling
        customer_features = self.cleaned_df.groupby('customer_id').agg({
            'age': 'first',
            'gender': 'first',
            'city': 'first',
            'loyalty_tier': 'first',
            'revenue': ['sum', 'mean', 'count'],
            'profit': 'sum',
            'quantity': 'mean',
            'discount_pct': 'mean',
            'returned': 'mean'
        })
        
        # Flatten column names
        customer_features.columns = ['_'.join(col).strip() for col in customer_features.columns.values]
        customer_features = customer_features.reset_index()
        
        # Feature engineering
        le = LabelEncoder()
        customer_features['gender_encoded'] = le.fit_transform(customer_features['gender_first'])
        customer_features['city_encoded'] = le.fit_transform(customer_features['city_first'])
        customer_features['loyalty_encoded'] = le.fit_transform(customer_features['loyalty_tier_first'])
        
        # Target variable: high value customer (top 20% by revenue)
        revenue_threshold = customer_features['revenue_sum'].quantile(0.8)
        customer_features['is_high_value'] = (customer_features['revenue_sum'] >= revenue_threshold).astype(int)
        
        # Select features for model
        feature_cols = ['age_first', 'gender_encoded', 'city_encoded', 'loyalty_encoded',
                       'revenue_mean', 'profit_sum', 'quantity_mean', 'discount_pct_mean', 'returned_mean']
        
        X = customer_features[feature_cols]
        y = customer_features['is_high_value']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate model
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop Predictive Features:")
        print(feature_importance.head())
        
        self.analysis_results['predictive_model'] = {
            'model': model,
            'features': feature_cols,
            'performance': {'mae': mae, 'r2': r2},
            'feature_importance': feature_importance
        }
        
        return model, feature_importance
    
    def generate_insights_and_recommendations(self):
        """
        Generate business insights and recommendations
        """
        print("\n" + "="*50)
        print("BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("="*50)
        
        insights = []
        recommendations = []
        
        # Extract key metrics
        total_revenue = self.analysis_results['exploratory_analysis']['total_revenue']
        return_rate = self.analysis_results['exploratory_analysis']['return_rate']
        category_perf = self.analysis_results['exploratory_analysis']['category_performance']
        customer_segments = self.analysis_results['customer_segmentation']
        
        # Generate insights
        insights.append(f"Total business revenue: ${total_revenue:,.2f}")
        insights.append(f"Overall return rate: {return_rate:.2%}")
        
        top_category = category_perf.index[0]
        insights.append(f"Top performing category: {top_category}")
        
        champion_count = len(customer_segments[customer_segments['segment'] == 'Champions'])
        insights.append(f"Number of champion customers: {champion_count}")
        
        # Generate recommendations
        if return_rate > 0.05:
            recommendations.append("Implement better quality control for high-return products")
        
        if category_perf.iloc[0]['revenue'] > category_perf.iloc[1]['revenue'] * 2:
            recommendations.append(f"Consider expanding {top_category} product line due to strong performance")
        
        if champion_count < len(customer_segments) * 0.1:
            recommendations.append("Develop loyalty programs to convert more customers to Champions segment")
        
        # Print insights and recommendations
        print("KEY INSIGHTS:")
        for insight in insights:
            print(f"  • {insight}")
        
        print("\nRECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  • {rec}")
        
        return insights, recommendations
    
    def run_complete_analysis(self):
        """
        Run the complete end-to-end analysis
        """
        print("E-COMMERCE ANALYTICS MINI PROJECT")
        print("="*60)
        
        # 1. Data generation
        self.generate_sample_data(n_customers=5000, n_products=200)
        
        # 2. Data quality assessment
        self.data_quality_check()
        
        # 3. Data cleaning
        self.clean_data()
        
        # 4. Exploratory analysis
        self.exploratory_analysis()
        
        # 5. Customer segmentation
        self.build_customer_segmentation()
        
        # 6. Predictive analytics
        self.predictive_analytics()
        
        # 7. Business insights
        self.generate_insights_and_recommendations()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)

if __name__ == "__main__":
    # Run the complete mini-project
    analytics = ECommerceAnalytics()
    analytics.run_complete_analysis()