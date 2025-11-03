"""
Easy 2: Solve a toy example applying Python Data Structures.
"""

class StudentManagementSystem:
    """A toy example demonstrating various Python data structures"""
    
    def __init__(self):
        # List: Ordered collection of students
        self.students = []
        
        # Dictionary: Student ID to details mapping
        self.student_details = {}
        
        # Set: Unique courses available
        self.available_courses = set()
        
        # Tuple: Fixed grading system
        self.grade_scale = ("A", "B", "C", "D", "F")
    
    def add_student(self, student_id, name, courses):
        """Add a student using multiple data structures"""
        # Add to list
        self.students.append(name)
        
        # Add to dictionary
        self.student_details[student_id] = {
            'name': name,
            'courses': courses,  # List of courses
            'grades': {}  # Dictionary of course to grade
        }
        
        # Add courses to set (automatically handles duplicates)
        self.available_courses.update(courses)
    
    def record_grade(self, student_id, course, grade):
        """Record a grade for a student"""
        if student_id in self.student_details:
            if course in self.available_courses:
                self.student_details[student_id]['grades'][course] = grade
                return True
        return False
    
    def get_student_gpa(self, student_id):
        """Calculate GPA using grade mapping"""
        grade_points = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'F': 0.0}
        
        if student_id in self.student_details:
            grades = self.student_details[student_id]['grades']
            if grades:
                total_points = sum(grade_points.get(grade, 0) for grade in grades.values())
                return total_points / len(grades)
        return 0.0
    
    def find_top_students(self, n=3):
        """Find top n students using sorting"""
        student_gpas = []
        for student_id, details in self.student_details.items():
            gpa = self.get_student_gpa(student_id)
            student_gpas.append((details['name'], gpa))
        
        # Sort by GPA descending
        student_gpas.sort(key=lambda x: x[1], reverse=True)
        return student_gpas[:n]
    
    def get_course_statistics(self):
        """Generate statistics for each course"""
        course_stats = {}
        
        for course in self.available_courses:
            grades = []
            for student_id, details in self.student_details.items():
                if course in details['grades']:
                    grades.append(details['grades'][course])
            
            if grades:
                grade_counts = {}
                for grade in grades:
                    grade_counts[grade] = grade_counts.get(grade, 0) + 1
                
                course_stats[course] = {
                    'total_students': len(grades),
                    'grade_distribution': grade_counts
                }
        
        return course_stats

def demonstrate_data_structures():
    """Demonstrate the toy example in action"""
    sms = StudentManagementSystem()
    
    # Add students with their courses
    sms.add_student(101, "Alice Johnson", ["Math", "Physics", "Chemistry"])
    sms.add_student(102, "Bob Smith", ["Math", "Biology", "History"])
    sms.add_student(103, "Carol Davis", ["Physics", "Chemistry", "Art"])
    sms.add_student(104, "David Wilson", ["Math", "Physics", "Computer Science"])
    
    # Record grades
    sms.record_grade(101, "Math", "A")
    sms.record_grade(101, "Physics", "B")
    sms.record_grade(102, "Math", "B")
    sms.record_grade(102, "Biology", "A")
    sms.record_grade(103, "Physics", "A")
    sms.record_grade(103, "Chemistry", "A")
    sms.record_grade(104, "Math", "C")
    sms.record_grade(104, "Physics", "B")
    
    # Display results
    print("STUDENT MANAGEMENT SYSTEM DEMONSTRATION")
    print("=" * 40)
    
    print(f"\nAll Students (List): {sms.students}")
    print(f"Available Courses (Set): {sms.available_courses}")
    print(f"Grade Scale (Tuple): {sms.grade_scale}")
    
    print(f"\nStudent Details (Dictionary):")
    for student_id, details in sms.student_details.items():
        print(f"  {student_id}: {details}")
    
    print(f"\nTop 3 Students:")
    for name, gpa in sms.find_top_students(3):
        print(f"  {name}: GPA {gpa:.2f}")
    
    print(f"\nCourse Statistics:")
    stats = sms.get_course_statistics()
    for course, stat in stats.items():
        print(f"  {course}: {stat['total_students']} students, Grades: {stat['grade_distribution']}")

if __name__ == "__main__":
    demonstrate_data_structures()