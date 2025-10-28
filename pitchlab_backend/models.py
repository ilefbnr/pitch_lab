from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime

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
    created_at: datetime
    user_id: int

    class Config:
        orm_mode = True
