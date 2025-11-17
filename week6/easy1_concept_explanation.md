# Advanced SQL Joins & Aggregations - Key Concepts

## Advanced Joins

### 1. Multiple Table Joins
Joining more than two tables in a single query to combine data from multiple sources.

### 2. Self Joins
Joining a table with itself, useful for hierarchical data or comparing rows within the same table.

### 3. Non-Equi Joins
Joins using comparison operators other than equality (=), such as <, >, BETWEEN.

### 4. Cross Joins
Cartesian product of two tables, generating all possible combinations.

### 5. Anti Joins
Finding records in one table that don't have matches in another table (using NOT EXISTS or LEFT JOIN + IS NULL).

## Advanced Aggregations

### 1. Window Functions
- **ROW_NUMBER()**: Assigns unique sequential integers
- **RANK()**: Assigns rank with gaps for ties
- **DENSE_RANK()**: Assigns rank without gaps for ties
- **LEAD()/LAG()**: Access subsequent/previous rows

### 2. GROUP BY Extensions
- **GROUPING SETS**: Multiple grouping levels in one query
- **CUBE**: All possible grouping combinations
- **ROLLUP**: Hierarchical grouping (subtotals)

### 3. Filtered Aggregations
Using FILTER clause or CASE statements within aggregate functions.

### 4. Statistical Aggregations
- **PERCENTILE_CONT**: Continuous percentiles
- **STDDEV**: Standard deviation
- **CORR**: Correlation coefficients