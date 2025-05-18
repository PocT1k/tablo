from pathlib import Path

# Базовая папка проекта (где лежит этот conf.py)
BASE_DIR = Path(__file__).parent.parent.resolve()

# Папка с данными, настройками, моделями и логами
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "model"
FACE_DIR = DATA_DIR / "face"
LOG_DIR = DATA_DIR / "log"

# Путь к файлу настроек
SETTING_JSON_PATH = DATA_DIR / "setting.json"

# Директории для обучения face-recognition
DATASET_RAW_DIR = FACE_DIR / "dataset"
DATASET_CONVERTED_DIR = FACE_DIR / "dataset_converted"

# Модели
# Папка зображений
IMAGE_MODEL_DIR = MODEL_DIR / "image"
# Распознование лица (Face)
FACE_MODEL_PATH = IMAGE_MODEL_DIR / "face_classifier.pkl"
FACE_MODEL_OLD_DIR = MODEL_DIR / "face_classifier_old"
# Распознование предметов (YOLOv5)
YOLO_MODEL_PATH = IMAGE_MODEL_DIR / "yolov5su.pt"
# Папка голоса
VOICE_MODEL_DIR = MODEL_DIR / "audio"
# Распознавание речи (Vosk)
VOSK_MODEL_PATH = VOICE_MODEL_DIR / "vosk-model-small-ru-0.22"
# Классификация звуков (YAMNet)
YAMNET_MODEL_PATH = VOICE_MODEL_DIR / "yamnet"

# Логи
FACE_ATTENDANCE_PATH = LOG_DIR / "face_attendance.csv"
FACE_ATTENDANCE_OLD_DIR = LOG_DIR / "face_old"
VOSK_WORLD_PATH = LOG_DIR / "vosk_world.csv"
VOSK_WORLD_OLD_DIR = LOG_DIR / "vosk_old"
YAMNET_INDICES_PATH = LOG_DIR / "yamn_indices.csv"
YAMNET_INDICES_OLD_DIR = LOG_DIR / "yamn_old"
YOLO_CLASSEC_PATH = LOG_DIR / "yolo_indices.csv"
YOLO_CLASSEC_OLD_DIR = LOG_DIR / "yolo_old"

# Копирование окружения
# pip freeze > data\requirements.txt
# pip install -r data\requirements.txt

# tablo/
# │
# ├── scripts/                   # Все Python-скрипты
# │   ├── main.py                # Точка входа
# │   ├── config.py              # Настройки (пути)
# │   ├── ui.py                  # Интерфейс (PyQt)
# │   ├── image.py               # Обработка изображение - распознавание лиц, детектирование предметов
# │   ├── audio.py               # Обработка звука - распознавание звуков, распознавание слов
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
