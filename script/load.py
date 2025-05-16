import os
import pickle
import shutil
from datetime import datetime
from typing import Any

from conf import face_model_path


def check_exist(file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

"""Получить или сдел зн. по умолчанию: (dict, ключ, зн. по умолч.)"""
def dict_get_or_set(dict_data: dict, key: str, default_value: Any, set_dict = True) -> Any:
    value = dict_data.get(key)

    if value is None: # Если ключ не найден
        print(f"[LOAD] Не найден ключ: '{key}', подставленно: {default_value}")
        if set_dict: dict_data[key] = default_value # изменяем dict если флаг
        value = default_value

    return value

def load_face_model():
    if os.path.exists(face_model_path):
        with open(face_model_path, "rb") as f:
            face_model = pickle.load(f)
        print("[LOAD] Модель распознования лиц загружена")
    else:
        face_model = None
        print("[LOAD] Модель распознования не обнаружена")

    return face_model


def archive_file(file_path: str, archive_dir: str, create_file = False):
    # Убедимся, что папка архива существует
    os.makedirs(archive_dir, exist_ok=True)

    # Разбиваем имя и расширение
    base, ext = os.path.splitext(os.path.basename(file_path))

    # Если исходный файл существует — переместим его
    if os.path.exists(file_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"{base}{timestamp}{ext}"
        dest_path = os.path.join(archive_dir, new_name)
        shutil.move(file_path, dest_path)
        print(f"[ARCHIVE] Перемещён из {file_path} в {dest_path}")

    # Создаём новый пустой файл
    if create_file:
        open(file_path, "w").close()
        print(f"[ARCHIVE] Создан новый файл {file_path}")

