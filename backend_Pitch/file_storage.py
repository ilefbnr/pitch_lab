import os
import uuid
import aiofiles
from pathlib import Path
from typing import Optional
import shutil
from dotenv import load_dotenv

load_dotenv()
class FileStorage:
    """
    File storage handler that can work with local storage or cloud storage
    """
    
    def __init__(self, storage_type: str = "local", base_path: str = "uploads"):
        self.storage_type = storage_type
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    async def save_audio_file(self, audio_content: bytes, filename: str) -> str:
        """
        Save audio file and return the file path
        """
        # Generate unique filename
        file_extension = filename.split('.')[-1] if '.' in filename else 'wav'
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = self.base_path / unique_filename
        print(file_path)
        if self.storage_type == "local":
            return await self._save_local(audio_content, file_path)
        elif self.storage_type == "cloud":
            # Future implementation for cloud storage (AWS S3, Google Cloud Storage, etc.)
            return await self._save_cloud(audio_content, unique_filename)
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")
    
    async def save_video_file(self, video_content: bytes, filename: str) -> str:
        """
        Save video file and return the file path
        """
        # Generate unique filename
        file_extension = filename.split('.')[-1] if '.' in filename else 'mp4'
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = self.base_path / unique_filename
        
        if self.storage_type == "local":
            return await self._save_local(video_content, file_path)
        elif self.storage_type == "cloud":
            return await self._save_cloud(video_content, unique_filename)
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")
    
    async def save_file(self, content: bytes, filename: str, file_type: str = "auto") -> str:
        """
        Generic file save method that auto-detects type or uses specified type
        """

        
        if file_type == "auto":
            # Auto-detect based on file extension
            if filename and '.' in filename:
                ext = filename.split('.')[-1].lower()
                if ext in ['mp3', 'wav', 'flac', 'aac', 'm4a']:
                    return await self.save_audio_file(content, filename)
                elif ext in ['mp4', 'avi', 'mov', 'webm', 'mkv']:
                    return await self.save_video_file(content, filename)
        elif file_type == "audio":
            return await self.save_audio_file(content, filename)
        elif file_type == "video":
            return await self.save_video_file(content, filename)
        
        # Default fallback - save as generic file
        file_extension = filename.split('.')[-1] if '.' in filename else 'bin'
        unique_filename = f"{uuid.uuid4()}.{file_extension}"

        file_path = self.base_path / unique_filename
        
        if self.storage_type == "local":
            return await self._save_local(content, file_path)
        else:
            return await self._save_cloud(content, unique_filename)
    
    async def _save_local(self, audio_content: bytes, file_path: Path) -> str:
        """Save file locally"""
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(audio_content)
        return str(file_path)
    
    async def _save_cloud(self, audio_content: bytes, filename: str) -> str:
        """
        Save file to cloud storage
        This is a placeholder - implement based on your cloud provider
        """
        # For now, save locally but return a cloud-like path
        local_path = await self._save_local(audio_content, self.base_path / filename)
        
        # In a real implementation, you would:
        # 1. Upload to AWS S3, Google Cloud Storage, Azure Blob, etc.
        # 2. Return the cloud URL
        # Example: return f"https://your-bucket.s3.amazonaws.com/{filename}"
        
        return local_path
    
    def get_file_url(self, file_path: str) -> str:
        """
        Get accessible URL for the file
        """
        if self.storage_type == "local":
            # For local files, return the path (in production, serve through a web server)
            return file_path
        else:
            # For cloud storage, return the public URL
            return file_path
    
    async def delete_file(self, file_path: str) -> bool:
        """
        Delete a file
        """
        try:
            if self.storage_type == "local":
                Path(file_path).unlink(missing_ok=True)
            else:
                # Implement cloud deletion
                pass
            return True
        except Exception:
            return False

# Initialize default storage
storage = FileStorage(
    storage_type=os.getenv("STORAGE_TYPE", "local"),
    base_path=os.getenv("UPLOAD_PATH", "uploads")
)
