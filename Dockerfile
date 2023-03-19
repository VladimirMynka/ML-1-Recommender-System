FROM python:3.10-slim

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN  echo "DockerBuildDate: $(date)" > credentials/docker_credentials.txt

CMD [ "gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "wsgi_app:application" ]
