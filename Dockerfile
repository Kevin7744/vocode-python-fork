FROM python:3.9-bullseye

RUN apt-get update  && \
    apt-get install libportaudio2 \
           libportaudiocpp0 portaudio19-dev \
           libasound-dev libsndfile1-dev -y

RUN apt-get -y update &&  \
    apt-get -y upgrade && \
    apt-get install -y ffmpeg


WORKDIR /code
COPY ./pyproject.toml /code/pyproject.toml

RUN pip install --no-cache-dir --upgrade poetry

RUN poetry config virtualenvs.create false

COPY vocode/ vocode/

RUN poetry install --no-dev --no-interaction --no-ansi

COPY apps/twilio_outbound/ /code/

RUN python -c 'import nltk; nltk.download("punkt")'