from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime
import uuid

class PitchBase(BaseModel):
    title: str
    description: Optional[str] = None

class PitchCreate(PitchBase):
    pass

class PitchResponse(PitchBase):
    id: uuid.UUID
    audio_url: Optional[str]
    video_url: Optional[str]
    transcript: Optional[str]
    analysis_result: Optional[Any]
    created_at: datetime

    class Config:
        orm_mode = True
