from fastapi import FastAPI, HTTPException, Query
from vad_logic import calculate_silence_segment
import tempfile
import requests
import os
import subprocess
import uuid
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/process")
@app.get("/process")
def process(
    url: str = Query(None, description="URL of the video file to process"),
    data: dict = None
):
    try:
        # Support both Query param and JSON body
        video_url = url
        if data and "url" in data:
            video_url = data["url"]
        
        if not video_url:
            raise HTTPException(status_code=400, detail="Missing 'url' parameter or body field")

        # 1. Calculate onset using Torch Logic
        silence_until = calculate_silence_segment(video_url)

        return {
            "silence_duration": round(silence_until, 3)
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
