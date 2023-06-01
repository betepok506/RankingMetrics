FROM python:3.9-slim

RUN mkdir -p docs
WORKDIR /app
COPY requirements-docs.txt /app
RUN pip3 install -r /app/requirements-docs.txt --no-cache-dir
COPY src/ /app/src/

CMD pdoc -d numpy -o docs src/docs_ranking_metrics/*.py && \
    chown -R $UID:$GID docs/*
