FROM nvcr.io/nvidia/tritonserver:23.12-py3

# Устанавливаем дополнительные зависимости при необходимости
# RUN apt-get update && apt-get install -y <дополнительные пакеты> && rm -rf /var/lib/apt/lists/*

# Скопируем модельный репозиторий и assets внутрь образа
COPY model_repository /models
COPY assets /assets

# Установим Python-зависимости для токенайзера
# Предполагается, что tokenizer/requirements.txt содержит необходимые пакеты, например transformers
RUN pip3 install --no-cache-dir -r /models/tokenizer/requirements.txt

# Убедимся, что LD_LIBRARY_PATH установлен (уже задаётся в окружении)
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Запустим Triton Inference Server
# Порты по умолчанию: HTTP=8000, GRPC=8001, Metrics=8002
ENTRYPOINT ["tritonserver", "--model-repository=/models", "--http-port=8000", "--grpc-port=8001", "--metrics-port=8002"]
