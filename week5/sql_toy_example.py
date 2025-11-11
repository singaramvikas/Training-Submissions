"""
Easy 2: Solve a toy example applying SQL Basics.

Let's create a simple student database and perform basic SQL operations.
We'll simulate common SQL queries using Python data structures.
"""

class ToySQLDatabase:
    def __init__(self):
        self.students = [
            {'id': 1, 'name': 'Alice', 'age': 20, 'grade': 'A', 'major': 'Computer Science'},
            {'id': 2, 'name': 'Bob', 'age': 22, 'grade': 'B', 'major': 'Mathematics'},
            {'id': 3, 'name': 'Charlie', 'age': 21, 'grade': 'A', 'major': 'Physics'},
            {'id': 4, 'name': 'Diana', 'age': 19, 'grade': 'C', 'major': 'Computer Science'},
            {'id': 5, 'name': 'Eve', 'age': 23, 'grade': 'B', 'major': 'Mathematics'}
        ]
    
    def select_all(self):
        """SELECT * FROM students"""
        return self.students
    
    def select_columns(self, columns):
        """SELECT column1, column2 FROM students"""
        return [{col: student[col] for col in columns} for student in self.students]
    
    def where_filter(self, condition):
        """SELECT * FROM students WHERE condition"""
        return [student for student in self.students if condition(student)]
    
    def group_by_major(self):
        """SELECT major, COUNT(*) FROM students GROUP BY major"""
        from collections import defaultdict
        grouped = defaultdict(int)
        for student in self.students:
            grouped[student['major']] += 1
        return dict(grouped)
    
    def order_by(self, key, descending=False):
        """SELECT * FROM students ORDER BY key"""
        return sorted(self.students, key=lambda x: x[key], reverse=descending)

def demonstrate_toy_example():
    db = ToySQLDatabase()
    
    print("TOY SQL EXAMPLE - STUDENT DATABASE")
    print("=" * 50)
    
    print("\n1. SELECT * FROM students:")
    for student in db.select_all():
        print(f"   {student}")
    
    print("\n2. SELECT name, major FROM students:")
    for student in db.select_columns(['name', 'major']):
        print(f"   {student}")
    
    print("\n3. SELECT * FROM students WHERE grade = 'A':")
    a_students = db.where_filter(lambda s: s['grade'] == 'A')
    for student in a_students:
        print(f"   {student}")
    
    print("\n4. SELECT major, COUNT(*) FROM students GROUP BY major:")
    majors_count = db.group_by_major()
    for major, count in majors_count.items():
        print(f"   {major}: {count} students")
    
    print("\n5. SELECT * FROM students ORDER BY age DESC:")
    sorted_students = db.order_by('age', descending=True)
    for student in sorted_students:
        print(f"   {student}")

if __name__ == "__main__":
    demonstrate_toy_example()