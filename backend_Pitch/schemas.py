from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class PitchCreate(BaseModel):
    title: str
    description: Optional[str] = ""

class PitchResponse(BaseModel):
    id: int
    title: str
    description: str
    transcript: str
    audio_file_path: Optional[str] = None
    video_file_path: Optional[str] = None  # Phase 4: Video file path
    analysis_result: Optional[str] = None  # JSON string of analysis results
    created_at: datetime
    user_id: int
    
    class Config:
        from_attributes = True
