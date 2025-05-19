from builtins import enumerate
import os
from conf import (
    FACE_ATTENDANCE_FILE, FACE_ATTENDANCE_OLD,
    VOSK_WORLD_FILE, VOSK_WORLD_OLD,
    YAMNET_INDICES_FILE,YAMNET_INDICES_OLD,
    YOLO_CLASSEC_FILE, YOLO_CLASSEC_OLD
)
from load import get_log_path


LOGS_FILES = [FACE_ATTENDANCE_FILE, VOSK_WORLD_FILE, YAMNET_INDICES_FILE, YOLO_CLASSEC_FILE]
LOGS_OLDS = [FACE_ATTENDANCE_OLD, VOSK_WORLD_OLD, YAMNET_INDICES_OLD, YOLO_CLASSEC_OLD]
TIME_REC = ["image_time_recognition", "image_time_recognition", "audio_time_recognition", "audio_time_recognition"]


def get_stats(setting, stat_name, stat_period):
    cont_logs = len(LOGS_FILES)
    if cont_logs != len(LOGS_OLDS) or cont_logs != len(TIME_REC):
        raise 'Ошибка stats.py - неверно указанны пути логирования, неправильначя длинна'

    # Определения путей и существования файлов
    paths = []
    for i, file in enumerate(LOGS_FILES):
        path, exists = get_log_path(file, LOGS_OLDS[i], stat_period[0])
        paths.append(exists)

    # Определение % собранной информации - будет в выводе
    for i in range (len(paths)):
        exists = os.path.exists(paths[i])

    # Анализ каджого файла отдельно

    # Сбор общего анализа

    # Итогова оценка - главный вывод


#TODO - весь анализ статистики