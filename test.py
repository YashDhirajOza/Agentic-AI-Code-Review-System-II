"""
Dummy program to test the Agentic AI Test Generation System
Contains various types of functions with different complexities and characteristics
"""

import hashlib
import json
import re
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Union


class UserManager:
    """User management system with authentication and authorization"""
    
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the user database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        conn.commit()
        conn.close()
    
    def hash_password(self, password: str) -> str:
        """Hash a password using SHA-256 (security-sensitive function)"""
        if not password:
            raise ValueError("Password cannot be empty")
        
        # Add salt for security
        salt = "secure_salt_2024"
        salted_password = password + salt
        
        return hashlib.sha256(salted_password.encode()).hexdigest()
    
    def validate_email(self, email: str) -> bool:
        """Validate email format using regex"""
        if not email:
            return False
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def create_user(self, username: str, password: str, email: str, role: str = "user") -> Dict[str, Union[bool, str]]:
        """Create a new user (authentication function with multiple validations)"""
        # Input validation
        if not username or len(username) < 3:
            return {"success": False, "error": "Username must be at least 3 characters"}
        
        if not password or len(password) < 8:
            return {"success": False, "error": "Password must be at least 8 characters"}
        
        if not self.validate_email(email):
            return {"success": False, "error": "Invalid email format"}
        
        if role not in ["user", "admin", "moderator"]:
            return {"success": False, "error": "Invalid role"}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if username already exists
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                return {"success": False, "error": "Username already exists"}
            
            # Check if email already exists
            cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
            if cursor.fetchone():
                return {"success": False, "error": "Email already exists"}
            
            # Hash password and create user
            password_hash = self.hash_password(password)
            cursor.execute("""
                INSERT INTO users (username, password_hash, email, role)
                VALUES (?, ?, ?, ?)
            """, (username, password_hash, email, role))
            
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            
            return {"success": True, "user_id": user_id, "message": "User created successfully"}
            
        except sqlite3.Error as e:
            return {"success": False, "error": f"Database error: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Union[int, str]]]:
        """Authenticate user login (security-sensitive function)"""
        if not username or not password:
            return None
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get user data
            cursor.execute("""
                SELECT id, username, password_hash, email, role, is_active
                FROM users WHERE username = ?
            """, (username,))
            
            user_data = cursor.fetchone()
            conn.close()
            
            if not user_data:
                return None
            
            user_id, db_username, stored_hash, email, role, is_active = user_data
            
            # Check if user is active
            if not is_active:
                return None
            
            # Verify password
            if stored_hash == self.hash_password(password):
                return {
                    "id": user_id,
                    "username": db_username,
                    "email": email,
                    "role": role
                }
            
            return None
            
        except sqlite3.Error:
            return None
        except Exception:
            return None
    
    def get_user_permissions(self, user_id: int) -> List[str]:
        """Get user permissions based on role (authorization function)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT role FROM users WHERE id = ? AND is_active = 1", (user_id,))
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return []
            
            role = result[0]
            
            # Define permissions based on role
            permissions = {
                "user": ["read_profile", "update_profile"],
                "moderator": ["read_profile", "update_profile", "moderate_content", "read_users"],
                "admin": ["read_profile", "update_profile", "moderate_content", "read_users", 
                         "create_users", "delete_users", "manage_system"]
            }
            
            return permissions.get(role, [])
            
        except Exception:
            return []


class DataProcessor:
    """Data processing utilities with various complexity levels"""
    
    @staticmethod
    def simple_calculator(a: float, b: float, operation: str) -> float:
        """Simple calculator function (low complexity)"""
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Division by zero")
            return a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    @staticmethod
    def factorial(n: int) -> int:
        """Calculate factorial (edge case testing)"""
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        
        if n == 0 or n == 1:
            return 1
        
        result = 1
        for i in range(2, n + 1):
            result *= i
        
        return result
    
    @staticmethod
    def process_json_data(json_string: str) -> Dict:
        """Process JSON data with error handling"""
        if not json_string:
            raise ValueError("JSON string cannot be empty")
        
        try:
            data = json.loads(json_string)
            
            if not isinstance(data, dict):
                raise ValueError("JSON must be an object")
            
            # Process the data
            processed_data = {}
            for key, value in data.items():
                if isinstance(value, str):
                    processed_data[key] = value.strip().lower()
                elif isinstance(value, (int, float)):
                    processed_data[key] = value
                elif isinstance(value, list):
                    processed_data[key] = [str(item).strip() for item in value]
                else:
                    processed_data[key] = str(value)
            
            return processed_data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error processing JSON: {str(e)}")
    
    @staticmethod
    def filter_and_sort_data(data: List[Dict], filter_key: str, filter_value: str, sort_key: str) -> List[Dict]:
        """Complex data filtering and sorting (high complexity)"""
        if not data:
            return []
        
        if not isinstance(data, list):
            raise TypeError("Data must be a list")
        
        filtered_data = []
        
        # Filter data
        for item in data:
            if not isinstance(item, dict):
                continue
            
            if filter_key in item:
                item_value = str(item[filter_key]).lower()
                if filter_value.lower() in item_value:
                    filtered_data.append(item)
        
        # Sort data
        try:
            if sort_key:
                filtered_data.sort(key=lambda x: x.get(sort_key, ''))
        except Exception:
            # If sorting fails, return unsorted data
            pass
        
        return filtered_data
    
    @staticmethod
    def generate_report(data: List[Dict], title: str = "Report") -> str:
        """Generate a text report from data (integration with side effects)"""
        if not data:
            return f"{title}\n{'='*len(title)}\nNo data available."
        
        report_lines = [title, "=" * len(title), ""]
        
        # Summary
        report_lines.append(f"Total records: {len(data)}")
        report_lines.append("")
        
        # Data details
        for i, item in enumerate(data, 1):
            report_lines.append(f"Record {i}:")
            for key, value in item.items():
                report_lines.append(f"  {key}: {value}")
            report_lines.append("")
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_lines.append(f"Generated on: {timestamp}")
        
        return "\n".join(report_lines)


class FileManager:
    """File management utilities with I/O operations"""
    
    @staticmethod
    def save_to_file(content: str, filename: str) -> bool:
        """Save content to file (I/O side effects)"""
        try:
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(content)
            return True
        except Exception:
            return False
    
    @staticmethod
    def read_from_file(filename: str) -> Optional[str]:
        """Read content from file (I/O side effects)"""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception:
            return None
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for security (security-sensitive)"""
        if not filename:
            return "unnamed_file"
        
        # Remove dangerous characters
        dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        sanitized = filename
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip('. ')
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = "unnamed_file"
        
        # Limit length
        if len(sanitized) > 255:
            sanitized = sanitized[:255]
        
        return sanitized


class NetworkUtils:
    """Network utility functions"""
    
    @staticmethod
    def validate_ip_address(ip: str) -> bool:
        """Validate IP address format"""
        if not ip:
            return False
        
        parts = ip.split('.')
        if len(parts) != 4:
            return False
        
        try:
            for part in parts:
                num = int(part)
                if num < 0 or num > 255:
                    return False
            return True
        except ValueError:
            return False
    
    @staticmethod
    def parse_url(url: str) -> Dict[str, str]:
        """Parse URL components (potential security issues)"""
        if not url:
            return {}
        
        # Simple URL parsing (intentionally basic for testing)
        result = {"original": url}
        
        # Check for protocol
        if "://" in url:
            protocol, rest = url.split("://", 1)
            result["protocol"] = protocol
        else:
            result["protocol"] = "http"
            rest = url
        
        # Extract domain and path
        if "/" in rest:
            domain, path = rest.split("/", 1)
            result["domain"] = domain
            result["path"] = "/" + path
        else:
            result["domain"] = rest
            result["path"] = "/"
        
        return result


# Utility functions with various characteristics
def is_prime(n: int) -> bool:
    """Check if a number is prime (mathematical function)"""
    if n < 2:
        return False
    
    if n == 2:
        return True
    
    if n % 2 == 0:
        return False
    
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    
    return True


def fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number (recursive potential)"""
    if n < 0:
        raise ValueError("Fibonacci is not defined for negative numbers")
    
    if n <= 1:
        return n
    
    # Use iterative approach for efficiency
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b


def validate_password_strength(password: str) -> Dict[str, Union[bool, str, int]]:
    """Validate password strength (security function)"""
    if not password:
        return {"valid": False, "score": 0, "message": "Password cannot be empty"}
    
    score = 0
    issues = []
    
    # Length check
    if len(password) >= 8:
        score += 2
    else:
        issues.append("Password must be at least 8 characters long")
    
    # Character variety checks
    if re.search(r'[a-z]', password):
        score += 1
    else:
        issues.append("Password must contain lowercase letters")
    
    if re.search(r'[A-Z]', password):
        score += 1
    else:
        issues.append("Password must contain uppercase letters")
    
    if re.search(r'\d', password):
        score += 1
    else:
        issues.append("Password must contain numbers")
    
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        score += 1
    else:
        issues.append("Password must contain special characters")
    
    # Bonus for length
    if len(password) >= 12:
        score += 1
    
    # Common password check (simplified)
    common_passwords = ["password", "123456", "qwerty", "abc123", "password123"]
    if password.lower() in common_passwords:
        score = 0
        issues = ["Password is too common"]
    
    is_valid = score >= 4 and len(issues) == 0
    
    return {
        "valid": is_valid,
        "score": score,
        "max_score": 6,
        "issues": issues,
        "message": "Password is strong" if is_valid else "Password needs improvement"
    }


if __name__ == "__main__":
    # Demo usage
    print("Dummy Program for AI Test Generation")
    print("=" * 40)
    
    # Test UserManager
    user_manager = UserManager()
    result = user_manager.create_user("testuser", "TestPass123!", "test@example.com")
    print(f"User creation result: {result}")
    
    # Test DataProcessor
    processor = DataProcessor()
    calc_result = processor.simple_calculator(10, 5, "add")
    print(f"Calculator result: {calc_result}")
    
    # Test utility functions
    print(f"Is 17 prime? {is_prime(17)}")
    print(f"10th Fibonacci: {fibonacci(10)}")
    
    # Test password validation
    pwd_result = validate_password_strength("MySecurePass123!")
    print(f"Password validation: {pwd_result}")