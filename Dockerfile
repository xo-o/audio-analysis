# Use official Python 3.9 slim image
FROM python:3.9-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (FFmpeg is critical for audio extraction)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
# Note: We use the --no-cache-dir flag to keep the image size smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY main.py vad_logic.py ./

# Expose the port FastAPI runs on
EXPOSE 8000

# Start the uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
