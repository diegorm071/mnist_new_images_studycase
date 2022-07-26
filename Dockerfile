FROM python:3.10.2-slim

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

EXPOSE 8080

WORKDIR /app

COPY ./app/ ./app
COPY ./.env ./.env
COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

CMD streamlit run --server.port 8080 app/streamlit_app.py