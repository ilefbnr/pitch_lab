from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    pitches = relationship("Pitch", back_populates="user")

class Pitch(Base):
    __tablename__ = "pitches"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text)
    audio_file_path = Column(String)
    video_file_path = Column(String)  # Phase 4: Video file path
    transcript = Column(Text)
    analysis_result = Column(Text)  # JSON string of ML analysis results
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Relationship
    user = relationship("User", back_populates="pitches")
