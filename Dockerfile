FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]