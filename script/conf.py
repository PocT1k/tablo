from pathlib import Path

# Базовая папка проекта (где лежит этот conf.py)
BASE_DIR = Path(__file__).parent.parent.resolve()

# Папка с данными, настройками, моделями и логами
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"
PEOPLE_DIR = DATA_DIR / "people"
LOG_DIR = DATA_DIR / "logs"

# Путь к файлу настроек
SETTING_JSON_PATH = DATA_DIR / "setting.json"

# Директории для обучения face-recognition
DATASET_RAW_DIR = PEOPLE_DIR / "dataset"
DATASET_CONVERTED_DIR = PEOPLE_DIR / "dataset_converted"

# Модели
# Папка зображений
IMAGE_MODEL_DIR = MODEL_DIR / "image"
# Распознование лица (Face)
FACE_MODEL_PATH = IMAGE_MODEL_DIR / "face.pkl"
FACE_MODEL_OLD_DIR = MODEL_DIR / "face_old"
# Распознование предметов (YOLOv5)
YOLO_MODEL_PATH = IMAGE_MODEL_DIR / "yolov5su.pt"
# Папка голоса
VOICE_MODEL_DIR = MODEL_DIR / "audio"
# Распознавание речи (Vosk)
VOSK_MODEL_PATH = VOICE_MODEL_DIR / "vosk-model-small-ru-0.22"
# Классификация звуков (YAMNet)
YAMNET_MODEL_PATH = VOICE_MODEL_DIR / "yamnet"

# Логи
FACE_ATTENDANCE_FILE = "face.csv"
FACE_ATTENDANCE_OLD = "face_old"
FACE_ATTENDANCE_PATH = LOG_DIR / FACE_ATTENDANCE_FILE
FACE_ATTENDANCE_OLD_DIR = LOG_DIR / FACE_ATTENDANCE_OLD

VOSK_WORLD_FILE = "vosk.csv"
VOSK_WORLD_OLD = "vosk_old"
VOSK_WORLD_PATH = LOG_DIR / VOSK_WORLD_FILE
VOSK_WORLD_OLD_DIR = LOG_DIR / VOSK_WORLD_OLD

YAMNET_INDICES_FILE = "yamn.csv"
YAMNET_INDICES_OLD = "yamn_old"
YAMNET_INDICES_PATH = LOG_DIR / YAMNET_INDICES_FILE
YAMNET_INDICES_OLD_DIR = LOG_DIR / YAMNET_INDICES_OLD

YOLO_CLASSEC_FILE = "yolo.csv"
YOLO_CLASSEC_OLD = "yolo_old"
YOLO_CLASSEC_PATH = LOG_DIR / YOLO_CLASSEC_FILE
YOLO_CLASSEC_OLD_DIR = LOG_DIR / YOLO_CLASSEC_OLD

# Копирование окружения
# pip freeze > data\requirements.txt
# pip install -r data\requirements.txt

# Обовить pip
# python -m pip install --upgrade pip
# pip --versuin
# должна быть 25.1.1+ для установки dlib

# Установка dlib для face-recognition
# pip install data/requ/dlib-19.24.99-cp312-cp312-win_amd64.whl
# или pip install dlib --no-binary=dlib

# Для dlib нужен cMake https://cmake.org/download/

# tablo/
# │
# ├── scripts/                   # Все Python-скрипты
# │   ├── main.py                # Точка входа
# │   ├── config.py              # Настройки (пути)
# │   ├── ui.py                  # Интерфейс (PyQt)
# │   ├── image.py               # Обработка изображение - распознавание лиц, детектирование предметов
# │   ├── audio.py               # Обработка звука - распознавание звуков, распознавание слов
# │   ├── stats.py               # Обработка статистики - открытие и чтение всех файлов
# │   └── load.py                # Вспомогательные функции
# │
# ├── data/                      # Все данные
# │   ├── faces/                 # Изображения лиц
# │   │   ├── dataset/           # Перед конвертацией(user1/, user2/)
# │   │   ├── dataset_converted/ # Уже сконвертированные изображения для переобучения
# │   ├── model/                 # Модели
# │   │   ├── audio/             # vosk, yamnet
# │   │   ├── image/             # face, yolo
# │   │   │   ├── face_old/     # Старые модели face
# │   ├── log/                   # Логи
# │   │   ├── face.csv
# │   │   ├── vosk.csv
# │   │   ├── yolo.csv
# │   │   ├── old/
# │   ├── requirements.txt       # Зависимости
# │   └── settings.json          # Настройки приложения
# │
# └── Dockerfile            # Для упаковки в Docker
