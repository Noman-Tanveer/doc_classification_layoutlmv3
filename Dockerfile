# syntax=docker/dockerfile:1

FROM python:3.10-slim

ENV PYHTONUNBUFFERED=1

RUN apt-get update \
  && apt-get -y install tesseract-ocr \
  && rm -rf /var/lib/apt/lists/*

RUN apt update \
  && apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt
COPY . .
ENTRYPOINT ["python3"]
CMD ["inference.py"]
