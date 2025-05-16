from pathlib import Path

# Базовая папка проекта (где лежит этот conf.py)
BASE_DIR = Path(__file__).parent.parent.resolve()

# Папка с данными, настройками, моделями и логами
DATA_DIR    = BASE_DIR / "data"
MODEL_DIR   = DATA_DIR / "model"
FACE_DIR    = DATA_DIR / "face"
LOG_DIR     = DATA_DIR / "log"

# Путь к файлу настроек
SETTING_JSON_PATH = DATA_DIR / "setting.json"

# Директории для обучения face-recognition
DATASET_RAW_DIR      = FACE_DIR / "dataset"
DATASET_CONVERTED_DIR= FACE_DIR / "dataset_converted"

# Модели
# Распознование лица
FACE_MODEL_PATH       = MODEL_DIR / "face" / "face_classifier.pkl"
FACE_MODEL_OLD_DIR    = MODEL_DIR / "face_old"
# Распознавание речи (Vosk)
VOSK_MODEL_PATH        = MODEL_DIR / "voice" / "vosk-model-small-ru-0.22"
# Классификация звуков (YAMNet)
YAMNET_MODEL_PATH      = MODEL_DIR / "voice" / "yamnet"

# Логи
FACE_ATTENDANCE_PATH      = LOG_DIR / "face_attendance.csv"
FACE_ATTENDANCE_OLD_DIR   = LOG_DIR / "face_old"
VOSK_WORLD_PATH           = LOG_DIR / "vosk_world.csv"
VOSK_WORLD_OLD_DIR        = LOG_DIR / "vosk_old"
YAMNET_INDICES_PATH       = LOG_DIR / "yamnet_indices.csv"
YAMNET_INDICES_OLD_DIR    = LOG_DIR / "yamnet_old"
