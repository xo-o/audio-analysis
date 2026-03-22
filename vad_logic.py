import os
import sys
import tempfile
import subprocess
import requests
import numpy as np
import json
import torch
import wave
import struct
import math
import boto3
from botocore.client import Config

# --- Configuration ---
# Use the same threshold and sample rate as the user's working script
SAMPLING_RATE = 16000
VAD_THRESHOLD = 0.5
LEAD_IN_BUFFER = 0.3 # 300ms
ENERGY_THRESHOLD_DB = -35.0

# Lazy-load VAD model in a global cache
_VAD_MODEL = None
_VAD_UTILS = None

def get_vad_model():
    global _VAD_MODEL, _VAD_UTILS
    if _VAD_MODEL is None:
        # Load from local or hub
        _VAD_MODEL, _VAD_UTILS = torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=False)
    return _VAD_MODEL, _VAD_UTILS

def get_rms_frames(wav_path, frame_ms=10):
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        spf = int(sr * frame_ms / 1000)
        results = []
        t = 0.0
        while True:
            raw = wf.readframes(spf)
            if len(raw) < spf * sw: break
            data = struct.unpack(f"<{spf}h", raw)
            rms = math.sqrt(sum(x*x for x in data) / spf)
            db = 20 * math.log10(rms / 32768.0) if rms > 0 else -100.0
            results.append((t, db))
            t += frame_ms / 1000.0
        return results

def get_speech_onset(wav_path):
    """Refined VAD + Strategy D logic using Torch."""
    model, utils = get_vad_model()
    (get_speech_timestamps, _, read_audio, _, _) = utils
    
    # 1. Read audio for VAD
    wav = read_audio(wav_path, sampling_rate=SAMPLING_RATE)
    
    # 2. Get high-level speech segments
    segs = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE, threshold=VAD_THRESHOLD, return_seconds=True)
    
    if not segs:
        # If no speech found, we don't silence anything (return 0.0)
        return 0.0
        
    win_start = segs[0]["start"]
    win_end = win_start + 2.0 # 2s window for Strategy D
    
    # 3. Refine with Strategy D (Energy search)
    frames = get_rms_frames(wav_path)
    window = [f for f in frames if win_start <= f[0] <= win_end]
    
    raw_onset = win_start
    for t, db in window:
        if db > ENERGY_THRESHOLD_DB:
            raw_onset = t
            break
            
    return max(0, raw_onset - LEAD_IN_BUFFER)

def apply_silence(input_path, output_path, onset):
    """Apply silence to the first 'onset' seconds of the video."""
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-af", f"volume=0:enable='between(t,0,{onset})'",
        "-c:v", "copy",
        output_path
    ], check=True, capture_output=True)

def calculate_silence_segment(input_path_or_url):
    """
    Calculate the duration of silence at the beginning of a video.
    Accepts a local file path or a URL.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # 1. Handle URL or local file
        if str(input_path_or_url).startswith("http"):
            video_path = os.path.join(temp_dir, "input_video.mp4")
            r = requests.get(input_path_or_url, stream=True)
            r.raise_for_status()
            with open(video_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        else:
            video_path = input_path_or_url
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"File not found: {video_path}")

        # 2. Extract 16kHz WAV for analysis
        wav_path = os.path.join(temp_dir, "analysis.wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-ar", str(SAMPLING_RATE), "-ac", "1", "-c:a", "pcm_s16le",
            wav_path
        ], check=True, capture_output=True)

        # 3. Detect onset using VAD + Strategy D
        onset = get_speech_onset(wav_path)
        
        return onset

    finally:
        # Cleanup temp directory
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error during cleanup: {e}")

def upload_to_r2(file_path, object_name):
    """Upload a file to Cloudflare R2 and return the public URL."""
    s3 = boto3.client(
        's3',
        endpoint_url=f"https://{os.getenv('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
        config=Config(signature_version='s3v4'),
        region_name='auto'
    )
    
    bucket = os.getenv('R2_BUCKET_NAME')
    s3.upload_file(file_path, bucket, object_name, ExtraArgs={'ContentType': 'video/mp4'})
    
    cdn_domain = os.getenv('R2_PUBLIC_DOMAIN')
    if cdn_domain.startswith('http'):
        return f"{cdn_domain}/{object_name}"
    return f"https://{cdn_domain}/{object_name}"
