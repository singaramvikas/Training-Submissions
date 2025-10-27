"""
Easy 2: Solve a toy example applying Pandas Fundamentals
"""

import pandas as pd
import numpy as np

def toy_example():
    """
    A simple toy example demonstrating basic Pandas operations
    """
    print("TOY EXAMPLE: Student Grades Analysis")
    print("=" * 40)
    
    # 1. Create a DataFrame from scratch
    data = {
        'Student': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'Math': [85, 92, 78, 88, 95],
        'Science': [90, 85, 92, 79, 88],
        'English': [78, 85, 90, 82, 91],
        'Attendance': [95, 88, 92, 85, 98]
    }
    
    df = pd.DataFrame(data)
    print("1. Original DataFrame:")
    print(df)
    print()
    
    # 2. Basic operations
    print("2. Basic Information:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print()
    
    # 3. Add calculated column
    df['Average'] = df[['Math', 'Science', 'English']].mean(axis=1).round(2)
    print("3. DataFrame with Average:")
    print(df)
    print()
    
    # 4. Filtering data
    high_achievers = df[df['Average'] > 85]
    print("4. High Achievers (Average > 85):")
    print(high_achievers)
    print()
    
    # 5. Sorting
    sorted_df = df.sort_values('Average', ascending=False)
    print("5. Students Sorted by Average:")
    print(sorted_df[['Student', 'Average']])
    print()
    
    # 6. Aggregation
    print("6. Subject-wise Statistics:")
    subject_stats = df[['Math', 'Science', 'English']].agg(['mean', 'max', 'min'])
    print(subject_stats)
    print()
    
    # 7. Handling missing data (creating a scenario)
    df_with_missing = df.copy()
    df_with_missing.loc[2, 'Science'] = np.nan
    df_with_missing.loc[4, 'Math'] = np.nan
    
    print("7. Handling Missing Data:")
    print("DataFrame with missing values:")
    print(df_with_missing)
    print("\nAfter filling missing values:")
    df_filled = df_with_missing.fillna(df_with_missing.mean())
    print(df_filled)

if __name__ == "__main__":
    toy_example()