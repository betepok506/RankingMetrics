#!/bin/bash
mkdir -p docs
docker compose -f ./docker-compose-docs.yaml up --build
