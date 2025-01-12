FROM python:3.12.8-slim-bookworm

RUN apt-get update

RUN mkdir /app

WORKDIR app/

CMD ["python", "app/app.py"]