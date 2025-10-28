from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(128), unique=True, nullable=False, index=True)
    email = Column(String(256), unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    pitches = relationship("Pitch", back_populates="user")

class Pitch(Base):
    __tablename__ = "pitches"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(256), nullable=False)
    description = Column(Text, nullable=True)
    audio_file_path = Column(String(1024), nullable=True)
    video_file_path = Column(String(1024), nullable=True)
    transcript = Column(Text, nullable=True)
    analysis_result = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    user = relationship("User", back_populates="pitches")