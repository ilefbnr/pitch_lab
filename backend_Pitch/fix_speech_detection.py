#!/usr/bin/env python3
"""
Script to fix speech detection issues
"""
import subprocess
import sys
import asyncio
import tempfile
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def install_missing_dependencies():
    """Install any missing dependencies"""
    dependencies = [
        'soundfile',
        'librosa',
        'whisper',
        'torch',
        'numpy'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} is available")
        except ImportError:
            print(f"❌ {dep} is missing, installing...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
                print(f"✅ {dep} installed successfully")
            except Exception as e:
                print(f"❌ Failed to install {dep}: {e}")

async def test_transcription_simple():
    """Simple transcription test"""
    print("\n🧪 Testing Whisper transcription directly...")
    
    try:
        import whisper
        import soundfile as sf
        import tempfile
        
        # Load Whisper model
        print("📥 Loading Whisper model...")
        model = whisper.load_model("base")
        print("✅ Whisper model loaded")
        
        # Create test audio (speech-like sine wave)
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Generate speech-like audio
        base_freq = 150
        audio = np.sin(2 * np.pi * base_freq * t)
        
        # Add harmonics to make it more speech-like
        audio += 0.3 * np.sin(2 * np.pi * base_freq * 2 * t)
        audio += 0.2 * np.sin(2 * np.pi * base_freq * 3 * t)
        
        # Add amplitude modulation
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
        audio = audio * modulation
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.7
        
        # Save to file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, audio, sample_rate)
            temp_path = temp_file.name
        
        print(f"🎵 Created test audio: {temp_path}")
        
        # Test transcription
        print("🎤 Testing Whisper transcription...")
        result = model.transcribe(temp_path, language="en")
        
        print(f"📝 Transcription result: '{result.get('text', 'No text')}'")
        print(f"📊 Language detected: {result.get('language', 'Unknown')}")
        
        # Clean up
        import os
        os.unlink(temp_path)
        
    except Exception as e:
        print(f"❌ Transcription test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    print("🔧 Speech Detection Fix Script")
    print("=" * 40)
    
    # Install dependencies
    print("1. Checking dependencies...")
    install_missing_dependencies()
    
    # Test transcription
    print("\n2. Testing transcription...")
    asyncio.run(test_transcription_simple())
    
    print("\n🎯 Recommendations:")
    print("1. Make sure your microphone is working and not muted")
    print("2. Speak clearly and loud enough for the microphone to pick up")
    print("3. Try speaking for at least 3-5 seconds at a time")
    print("4. Check that your browser has microphone permissions")
    print("5. The system now has more lenient speech detection thresholds")

if __name__ == "__main__":
    main()
