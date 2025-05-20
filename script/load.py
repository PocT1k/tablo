import os
import re
import shutil
from datetime import datetime, date
from typing import Any

from conf import LOG_DIR


"""Получить или сдел зн. по умолчанию: (dict, ключ, зн. по умолч.)"""
def dict_get_or_set(dict_data: dict, key: str, default_value: Any = None, save = True) -> Any:
    if dict_data == {}:
        if default_value is None:
            print(f"[LOAD] Словарь пустой, а дефолтного значения нет. Ключ '{key}'")
            exit(0)
        else:
            dict_data[key] = default_value

    value = dict_data.get(key)
    if value is None:
        if default_value:
            value = default_value
            print(f"[LOAD] Не найден ключ: '{key}', подставленно: {default_value}")
            if save: # изменяем dict если флаг
                dict_data[key] = default_value
    return value

def archive_file(file_path: str, archive_dir: str, create_file = False):
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

# Архивировать файл, если он не сегодняшний
def archive_file_by_date(file_path: str, archive_dir: str, create_file: bool = False):
    # Папка, где лежат файлы
    folder = os.path.dirname(file_path)
    base, ext = os.path.splitext(os.path.basename(file_path))

    today = datetime.now().strftime("%Y%m%d")
    os.makedirs(archive_dir, exist_ok=True)

    # Шаблон: base + 8 цифр даты + ext
    pattern = re.compile(rf"^{re.escape(base)}(\d{{8}}){re.escape(ext)}$")
    # Проходим по файлам в папке
    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if not m:
            continue
        file_date = m.group(1)
        src = os.path.join(folder, fname)
        # Если дата не сегодняшняя — архивируем
        if file_date != today:
            dst = os.path.join(archive_dir, fname)
            try:
                shutil.move(src, dst)
                print(f"[DATA ARCHIVE] Перемещён {src} -> {dst}")
            except Exception as e:
                print(f"[DATA ARCHIVE ERROR] Не удалось переместить {src}: {e}")
        else:
            print(f"[DATA ARCHIVE] Оставлен {src}")

    # Создаём новый файл для сегодняшней даты
    if create_file:
        today_fname = f"{base}{today}{ext}"
        today_path = os.path.join(folder, today_fname)
        # Если файла нет
        if not os.path.exists(today_path):
            try:
                with open(today_path, "w", encoding="cp1251") as f:
                    pass
                print(f"[DATA ARCHIVE] Создан новый файл {today_path}")
            except Exception as e:
                print(f"[DATA ARCHIVE ERROR] Не удалось создать {today_path}: {e}")

def write_attendance_dated(file_path: str, text, timestamp = None, who_logging: str = '[]'):
    now = datetime.now()
    if not timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d,%H:%M:%S")
    date_tag = now.strftime("%Y%m%d")
    folder, fname = os.path.split(file_path)
    base, ext = os.path.splitext(fname)
    # Формируем
    dated_name = f"{base}{date_tag}{ext}"
    dated_path = os.path.join(folder, dated_name)
    os.makedirs(folder, exist_ok=True)

    if not text:
        text = 'None'
    elif isinstance(text, (list, tuple)): # Соединяем элементы списка запятыми, если массив
        text = ",".join(str(item) for item in text)
    else: # Экранируем запятые в строке, если строка
        text = text.replace(",", " ")

    with open(dated_path, "a", encoding="cp1251") as f:
        f.write(f"{timestamp},{text}\n")
    print(f"{who_logging} {timestamp} - {text}")
    return dated_path

def get_log_path(file_name: str, old_path: str, passed_date: date) -> (str, bool):
    # разбиваем имя и расширение
    base, ext = os.path.splitext(file_name)
    date_str = passed_date.strftime("%Y%m%d")
    dated_name = f"{base}{date_str}{ext}"

    if passed_date == date.today():
        folder = LOG_DIR
    else:
        folder = os.path.join(LOG_DIR, old_path)

    full_path = os.path.join(folder, dated_name)
    exists = os.path.exists(full_path)
    return full_path, exists
