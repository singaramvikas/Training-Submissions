"""
Easy 1: Describe the key concept of SQL Basics in your own words.

SQL (Structured Query Language) Basics can be understood through these core concepts:

1. TABLES: Data is organized in tables with rows (records) and columns (attributes)
2. CRUD OPERATIONS:
   - CREATE: Make new tables or databases
   - READ: Retrieve data using SELECT queries
   - UPDATE: Modify existing data
   - DELETE: Remove data

3. KEY CLAUSES:
   - SELECT: Choose which columns to retrieve
   - FROM: Specify which table to query
   - WHERE: Filter rows based on conditions
   - GROUP BY: Aggregate data by specific columns
   - HAVING: Filter aggregated results
   - ORDER BY: Sort the results

4. RELATIONAL OPERATIONS:
   - JOIN: Combine data from multiple tables
   - UNION: Combine results from multiple queries

In essence, SQL provides a standardized way to interact with relational databases,
allowing you to store, retrieve, and manipulate structured data efficiently.
"""

def explain_sql_basics():
    concepts = {
        "Core Purpose": "Manage and query structured data in relational databases",
        "Main Operations": "SELECT (read), INSERT (create), UPDATE (modify), DELETE (remove)",
        "Key Components": [
            "Tables with rows and columns",
            "Primary keys for unique identification",
            "Foreign keys for table relationships",
            "Indexes for performance optimization"
        ],
        "Basic Query Structure": "SELECT columns FROM table WHERE conditions GROUP BY columns ORDER BY columns"
    }
    
    print("SQL BASICS - KEY CONCEPTS")
    print("=" * 40)
    for key, value in concepts.items():
        print(f"\n{key}:")
        if isinstance(value, list):
            for item in value:
                print(f"  • {item}")
        else:
            print(f"  {value}")

if __name__ == "__main__":
    explain_sql_basics()