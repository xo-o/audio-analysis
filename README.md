# Audio Analysis Service (VAD)

This service provides robust Voice Activity Detection (VAD) and silence application for UGC videos. It uses the Silero VAD model (via PyTorch) and a custom energy-based refinement (Strategy D) to accurately identify the start of speech in a video and silence any preceding audio.

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

3.  **Configure Environment**:
    Create a `.env` file in the root directory with the following variables:
    ```env
    R2_ACCOUNT_ID=your_account_id
    R2_ACCESS_KEY_ID=your_access_key
    R2_SECRET_ACCESS_KEY=your_secret_key
    R2_BUCKET_NAME=your_bucket_name
    R2_PUBLIC_DOMAIN=your_public_domain
    ```

## Running the Service

Start the FastAPI server using the Python 3.9 environment:

```bash
./venv39/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
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
      "silence_until": 0.9
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
3.  **Buffer**: Subtracts a 300ms `LEAD_IN_BUFFER` to ensure natural starts.
4.  **Silence Application**: Uses FFmpeg's `volume=0` filter to silence the calculated duration.

## Repository Structure

-   `main.py`: FastAPI server and API endpoints.
-   `vad_logic.py`: Core detection, silence application, and R2 upload logic.
-   `requirements.txt`: Python dependencies.
-   `.gitignore`: Excludes environment files, large binaries, and the virtual environment.
