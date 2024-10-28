# Используем официальный образ Ubuntu версии 20.04
FROM ubuntu:20.04

# Устанавливаем неинтерактивный режим для apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Обновляем пакеты и устанавливаем необходимые зависимости
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

# Обновляем pip и устанавливаем инструменты для сборки Python пакетов
RUN pip3 install --upgrade pip
RUN pip3 install build

# Копируем ваш проект в контейнер
WORKDIR /app
COPY . /app

# Собираем ваш пакет
RUN python3 -m build

# Устанавливаем сгенерированный .whl файл
RUN pip3 install dist/*.whl

# Устанавливаем numpy для тестов
RUN pip3 install numpy

# Указываем, что контейнер будет запускаться в интерактивном режиме
CMD ["/bin/bash"]
