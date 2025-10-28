from supabase import create_client, Client
from dotenv import load_dotenv
import os
import uuid
import aiofiles

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "pitch-files")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class SupabaseStorage:
    async def save_audio_file(self, file_bytes, filename):
        unique_name = f"{uuid.uuid4()}_{filename}"
        res = supabase.storage.from_(SUPABASE_BUCKET).upload(unique_name, file_bytes)
        if res.get("error"):
            raise Exception(res["error"]["message"])
        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(unique_name)
        return public_url

    async def save_video_file(self, file_bytes, filename):
        unique_name = f"{uuid.uuid4()}_{filename}"
        res = supabase.storage.from_(SUPABASE_BUCKET).upload(unique_name, file_bytes)
        if res.get("error"):
            raise Exception(res["error"]["message"])
        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(unique_name)
        return public_url

storage = SupabaseStorage()
