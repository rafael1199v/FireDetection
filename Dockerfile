FROM python:3.14.0-alpine3.22

RUN apk update
RUN apk add gdal-dev
RUN apk add build-base
RUN apk add proj-dev
RUN apk add geos-dev
RUN apk add g++

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python","-u", "main.py"]