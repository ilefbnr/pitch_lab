from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime

class UserCreate(BaseModel):
    username: str
    email: Optional[str] = None

class UserLogin(BaseModel):
    username: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: Optional[str]
    created_at: Optional[datetime]
    class Config:
        orm_mode = True

class PitchCreate(BaseModel):
    title: str
    description: Optional[str] = None

class PitchResponse(BaseModel):
    id: int
    title: str
    description: Optional[str]
    transcript: Optional[str]
    audio_file_path: Optional[str]
    video_file_path: Optional[str]
    analysis_result: Optional[Any]
    created_at: Optional[datetime]
    user_id: Optional[int]
    class Config:
        orm_mode = True