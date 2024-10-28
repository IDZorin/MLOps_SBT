FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Устанавливаем необходимые зависимости
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-pybind11 \
    libopenblas-dev \
    git

RUN pip3 install --upgrade pip
RUN pip3 install build

# Копируем все файлы проекта в контейнер
WORKDIR /app
COPY . /app

# Собираем проект
RUN python3 -m build

# Устанавливаем пакет
RUN pip3 install dist/*.whl

# Устанавливаем numpy для тестов
RUN pip3 install numpy

CMD ["/bin/bash"]
