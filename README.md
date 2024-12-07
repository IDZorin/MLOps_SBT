# HW 1 - Биндинг Линейная Алгебра

## Обзор

**Линейная Алгебра** с использованием C++ и **pybind11** для создания биндингов Python.

Возможности:

- **Транспонирование**: Транспонирование матрицы.

## Команды Docker

### a. Вход в Docker Hub

Перед сборкой и публикацией Docker образов, необходимо войти в Docker Hub.

```sh
docker login
```

### b. Сборка Docker образа

Собрать Docker образ без использования кэша (с последними изменениями):

```sh
docker build --no-cache -t linalg_image .
```

Собрать Docker образ с кэшем:

```sh
docker build -t linalg_image .
```

### c. Запуск тестов внутри Docker контейнера

Тесты:

```sh
docker run --rm linalg_image python3 test_script.py
```

`--rm` для автоматического удаления контейнера после завершения работы.

### Результаты тестов

```plaintext
-- TRANSPOSE TEST --

Initial:
[1.0, 2.0, 3.0]
[4.0, 5.0, 6.0]

LinearAlgebra.T:
[1.0, 4.0]
[2.0, 5.0]
[3.0, 6.0]

NumPy.T:
[1.0, 4.0]
[2.0, 5.0]
[3.0, 6.0]

LinearAlgebra: 4.18
NumPy: 2.21

Diff: 1.90 times
```

# HW 2 - **Hydra, DVC, Lightning**

## Краткое описание выполненных шагов

1. **Подготовка DataModule и Model**:

   - Использован Python модуль с биндингами для подготовки данных в `TitanicDataModule`.
   - Реализован `SimpleTitanicModel` для обучения модели.

2. **Hydra конфигурация**:

   - Созданы конфигурационные файлы для Hydra (`config.yaml`, `data.yaml`, `model.yaml`, `trainer.yaml`).
   - Точка входа `train.py` использует Hydra для управления конфигурациями.

3. **DVC интеграция**:

   - Подключен DVC для управления данными.
   - Добавлены данные в DVC и настроено удаленное хранилище (через onedrive при помощи rclone).

4. **Запуск обучения**:

   - Обучение модели запускается с использованием подготовленных конфигураций и данных.

## Команды

**rclone**

```
./rclone.exe mount onedrive:dvc-storage T: --vfs-cache-mode writes
```

**dvc**

```
dvc add dvc_data/titanic.csv
dvc commit
```

**data collection**

```
python scripts\generate_titanic.py
```

**training**

```
python src/train.py
```
