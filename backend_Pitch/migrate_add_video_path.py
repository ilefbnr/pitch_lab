#!/usr/bin/env python3
"""
Migration script to add video_file_path column to pitches table
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import engine
from models import Base
from sqlalchemy import text

def migrate_add_video_path():
    """Add video_file_path column to pitches table"""
    try:
        # Check if column already exists
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'pitches' 
                AND column_name = 'video_file_path'
            """))
            
            if result.fetchone():
                print("✅ video_file_path column already exists")
                return
        
        # Add the column
        with engine.connect() as conn:
            conn.execute(text("""
                ALTER TABLE pitches 
                ADD COLUMN video_file_path VARCHAR
            """))
            conn.commit()
        
        print("✅ Successfully added video_file_path column to pitches table")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        # Fallback: recreate tables
        try:
            print("🔄 Attempting to recreate tables...")
            Base.metadata.drop_all(bind=engine)
            Base.metadata.create_all(bind=engine)
            print("✅ Tables recreated successfully")
        except Exception as e2:
            print(f"❌ Table recreation failed: {e2}")

if __name__ == "__main__":
    print("🔄 Running migration to add video_file_path column...")
    migrate_add_video_path()
    print("✅ Migration completed!") 