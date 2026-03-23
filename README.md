# Audio Analysis Service (VAD)

This service provides robust Voice Activity Detection (VAD) for UGC videos. It uses the Silero VAD model (via PyTorch) and a custom energy-based refinement (Strategy D) to accurately identify the start of speech in a video.

## Prerequisites

- **Python 3.9.6** (Standard on macOS)
- **FFmpeg** (Required for audio extraction)

## Installation

1.  **Create a Virtual Environment**:
    ```bash
    /usr/bin/python3 -m venv venv39
    source venv39/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install torch torchaudio requests numpy python-dotenv boto3 botocore fastapi uvicorn
    ```

## Running the Service

Start the FastAPI server using the Python 3.9 environment:

```bash
./venv39/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## Deployment

### 1. Railway (Easiest)
Railway will automatically detect the `Dockerfile` in the root directory.
-   Connect your GitHub repository.
-   Railway will build and deploy the container.
-   No manual `ffmpeg` or `torch` setup required.

### 2. Google Cloud Run (Scalable)
Build and push the Docker image to Container Registry:
```bash
gcloud builds submit --tag gcr.io/your-project/audio-vad
gcloud run deploy --image gcr.io/your-project/audio-vad --platform managed
```

### 3. Manual Docker Build
If you want to run the container locally:
```bash
docker build -t audio-vad .
docker run -p 8000:8000 audio-vad
```

## API Documentation

### 1. Process Video
Calculates the silence duration at the beginning of a video.

-   **URL**: `/process`
-   **Method**: `POST` (Recommended) or `GET`
-   **Body (JSON)**:
    ```json
    {
      "url": "https://example.com/video.mp4"
    }
    ```
-   **Query Parameters (GET)**:
    -   `url`: The video URL to process.

-   **Response**:
    ```json
    {
      "silence_until": 1.2
    }
    ```
    -   `silence_until`: The exact duration (in seconds) of silence at the beginning of the clip.

### 2. Health Check
Returns the current status of the service.

-   **URL**: `/health`
-   **Method**: `GET`
-   **Response**: `{"status": "ok"}`

## Core Logic (VAD + Strategy D)

1.  **VAD Detection**: Uses Silero VAD (v5) at 16kHz to find the first sustained speech segment.
2.  **Strategy D (Refinement)**: Searches for the exact point where audio energy crosses -35dB within a 2-second window starting from the VAD onset.
3.  **Result**: Returns the exact raw onset time (in seconds). The client is responsible for any leading offsets or buffers.

## Repository Structure

-   `main.py`: FastAPI server and API endpoints.
-   `vad_logic.py`: Core detection logic using PyTorch and Strategy D refinement.
-   `requirements.txt`: Python dependencies.
-   `.gitignore`: Excludes environment files and the virtual environment.
