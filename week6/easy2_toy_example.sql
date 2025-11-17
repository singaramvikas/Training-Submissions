-- Advanced SQL Joins & Aggregations - Toy Example

-- Sample tables
CREATE TABLE employees (
    emp_id INT PRIMARY KEY,
    name VARCHAR(50),
    department_id INT,
    salary DECIMAL(10,2),
    manager_id INT
);

CREATE TABLE departments (
    dept_id INT PRIMARY KEY,
    dept_name VARCHAR(50)
);

CREATE TABLE sales (
    sale_id INT PRIMARY KEY,
    emp_id INT,
    sale_date DATE,
    amount DECIMAL(10,2)
);

-- Insert sample data
INSERT INTO departments VALUES 
(1, 'Sales'), (2, 'Marketing'), (3, 'IT');

INSERT INTO employees VALUES 
(1, 'John Doe', 1, 50000, NULL),
(2, 'Jane Smith', 1, 60000, 1),
(3, 'Bob Johnson', 2, 55000, 1),
(4, 'Alice Brown', 3, 70000, NULL);

INSERT INTO sales VALUES 
(1, 2, '2024-01-15', 1000),
(2, 2, '2024-01-20', 1500),
(3, 3, '2024-01-25', 800),
(4, 2, '2024-02-01', 2000);

-- Advanced Join: Multiple tables with aggregation
SELECT 
    d.dept_name,
    e.name AS employee_name,
    COUNT(s.sale_id) AS total_sales,
    SUM(s.amount) AS total_revenue,
    AVG(s.amount) AS avg_sale_amount
FROM departments d
INNER JOIN employees e ON d.dept_id = e.department_id
LEFT JOIN sales s ON e.emp_id = s.emp_id
GROUP BY d.dept_name, e.name
ORDER BY total_revenue DESC;

-- Self Join: Employees and their managers
SELECT 
    e.name AS employee,
    m.name AS manager,
    e.salary AS employee_salary,
    m.salary AS manager_salary
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.emp_id;

-- Window Function: Rank employees by salary within department
SELECT 
    name,
    department_id,
    salary,
    RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) as salary_rank,
    AVG(salary) OVER (PARTITION BY department_id) as avg_dept_salary
FROM employees;

-- Advanced Aggregation with Filtering
SELECT 
    d.dept_name,
    COUNT(DISTINCT e.emp_id) AS total_employees,
    COUNT(s.sale_id) AS total_sales,
    SUM(CASE WHEN s.amount > 1000 THEN s.amount ELSE 0 END) AS high_value_sales,
    AVG(s.amount) FILTER (WHERE s.amount > 0) AS avg_non_zero_sale
FROM departments d
LEFT JOIN employees e ON d.dept_id = e.department_id
LEFT JOIN sales s ON e.emp_id = s.emp_id
GROUP BY d.dept_name;