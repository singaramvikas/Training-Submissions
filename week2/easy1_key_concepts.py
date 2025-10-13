"""
Easy 1: Describe the key concept of Python Data Structures in your own words.
"""

def explain_data_structures():
    """
    Python Data Structures are containers that organize and store data in specific ways
    to enable efficient operations. Each structure has unique characteristics that make
    it suitable for different scenarios.
    """
    
    concepts = {
        "Lists": {
            "description": "Ordered, mutable sequences that allow duplicate elements",
            "key_features": [
                "Maintains insertion order",
                "Elements accessible by index",
                "Can contain mixed data types",
                "Dynamic resizing"
            ],
            "common_operations": ["append", "insert", "remove", "slice"],
            "time_complexity": "O(1) for append/pop, O(n) for insert/remove"
        },
        
        "Tuples": {
            "description": "Ordered, immutable sequences",
            "key_features": [
                "Fixed once created",
                "Faster than lists",
                "Can be used as dictionary keys",
                "Memory efficient"
            ],
            "common_operations": ["count", "index", "unpacking"],
            "time_complexity": "O(1) for access"
        },
        
        "Dictionaries": {
            "description": "Unordered key-value pairs with fast lookup",
            "key_features": [
                "Key-based access (like hash maps)",
                "No duplicate keys allowed",
                "Mutable but keys must be hashable",
                "Python 3.7+ maintains insertion order"
            ],
            "common_operations": ["get", "keys", "values", "items"],
            "time_complexity": "O(1) for lookup/insert/delete"
        },
        
        "Sets": {
            "description": "Unordered collections of unique elements",
            "key_features": [
                "No duplicate elements",
                "Mathematical set operations",
                "Elements must be hashable",
                "Very fast membership testing"
            ],
            "common_operations": ["union", "intersection", "difference"],
            "time_complexity": "O(1) for membership test"
        }
    }
    
    print("PYTHON DATA STRUCTURES - KEY CONCEPTS")
    print("=" * 50)
    
    for structure, info in concepts.items():
        print(f"\n{structure}:")
        print(f"  Description: {info['description']}")
        print(f"  Key Features: {', '.join(info['key_features'])}")
        print(f"  Common Operations: {', '.join(info['common_operations'])}")
        print(f"  Time Complexity: {info['time_complexity']}")

if __name__ == "__main__":
    explain_data_structures()