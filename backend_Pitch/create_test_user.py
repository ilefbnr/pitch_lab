#!/usr/bin/env python3
"""
Script to create a test user for the Joy application
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import get_db, engine
from models import User, Base
from auth import get_password_hash

def create_test_user():
    """Create a test user for easy access"""
    
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    
    # Get database session
    db = next(get_db())
    
    try:
        # Check if test user already exists
        existing_user = db.query(User).filter(User.email == "test@joy.com").first()
        
        if existing_user:
            print("✅ Test user already exists:")
            print(f"   Email: test@joy.com")
            print(f"   Username: {existing_user.username}")
            print(f"   Password: test123")
            return
        
        # Create test user
        test_user = User(
            email="test@joy.com",
            username="testuser",
            hashed_password=get_password_hash("test123")
        )
        
        db.add(test_user)
        db.commit()
        
        print("✅ Test user created successfully!")
        print(f"   Email: test@joy.com")
        print(f"   Username: testuser")
        print(f"   Password: test123")
        print("\n🔑 You can now login with these credentials")
        
    except Exception as e:
        print(f"❌ Error creating test user: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    create_test_user() 