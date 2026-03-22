from fastapi import FastAPI, HTTPException, Query
from vad_logic import calculate_silence_segment, apply_silence, upload_to_r2
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

        # 1. Download to temp file (Needed for apply_silence)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
            r = requests.get(video_url, stream=True)
            r.raise_for_status()
            for chunk in r.iter_content(8192):
                tmp_video.write(chunk)
            tmp_video_path = tmp_video.name

        # 2. Calculate onset using Torch Logic
        # (This internally handles wav extraction)
        silence_until = calculate_silence_segment(video_url)

        # 3. Apply silence
        if silence_until > 0:
            silenced_path = tmp_video_path + "_silenced.mp4"
            apply_silence(tmp_video_path, silenced_path, silence_until)
        else:
            silenced_path = tmp_video_path

        # 4. Upload to R2
        object_name = f"processed/ugc-{uuid.uuid4()}.mp4"
        public_url = upload_to_r2(silenced_path, object_name)

        # 5. Cleanup
        os.remove(tmp_video_path)
        if silence_until > 0:
            os.remove(silenced_path)

        return {
            "url": public_url,
            "silence_duration": round(silence_until, 3)
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
