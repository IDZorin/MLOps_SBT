#!/bin/bash
set -e

# Устанавливаем зависимости из requirements.txt для модели tokenizer
pip install -r /models/tokenizer/requirements.txt

# Запускаем Triton Inference Server с репозиторием моделей
exec tritonserver --model-repository=/models --http-port=8000 --grpc-port=8001 --metrics-port=8002