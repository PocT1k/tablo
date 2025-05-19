from builtins import enumerate
from datetime import datetime, date, time, timedelta
import csv
import os

from conf import (
    FACE_ATTENDANCE_FILE, FACE_ATTENDANCE_OLD,
    VOSK_WORLD_FILE, VOSK_WORLD_OLD,
    YAMNET_INDICES_FILE,YAMNET_INDICES_OLD,
    YOLO_CLASSEC_FILE, YOLO_CLASSEC_OLD,
)
from load import get_log_path, dict_get_or_set


LOGS_FILES = [
    FACE_ATTENDANCE_FILE,
    YOLO_CLASSEC_FILE,
    VOSK_WORLD_FILE,
    YAMNET_INDICES_FILE
]
LOGS_OLDS = [
    FACE_ATTENDANCE_OLD,
    YOLO_CLASSEC_OLD,
    VOSK_WORLD_OLD,
    YAMNET_INDICES_OLD
]
# ключи в setting.json для интервалов, в тех же порядках
TIME_KEYS = [
    "image_time_recognition",
    "image_time_recognition",
    "audio_time_recognition",
    "audio_time_recognition"
]


def compute_log_percentages(setting: dict, date_obj: date, dt_start: datetime, dt_end: datetime):
    n = len(LOGS_FILES)
    percs = []
    details = []

    for i, fname in enumerate(LOGS_FILES):
        old_dir = LOGS_OLDS[i]
        interval = setting.get(TIME_KEYS[i], None)
        if interval is None or interval <= 0:
            raise ValueError(f"Не задан или некорректен интервал {TIME_KEYS[i]}")

        path, exists = get_log_path(fname, old_dir, date_obj)
        if not exists:
            percs.append(0.0)
            details.append((fname, path, 0, 0, 0.0))
            continue

        count = 0
        with open(path, newline="", encoding="cp1251", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    ts = datetime.strptime(row[0] + "," + row[1], "%Y-%m-%d,%H:%M:%S")
                except ValueError:
                    continue
                if dt_start <= ts <= dt_end:
                    count += 1

        period_sec = (dt_end - dt_start).total_seconds()
        expected = int(max(1.0, period_sec / interval))
        base_frac = count / expected
        coef = dict_get_or_set(setting, "stats_persent_coef", 1.02)
        adj_frac = min(base_frac * coef, 1.0)
        pct = adj_frac * 100.0

        percs.append(pct)
        details.append((fname, path, count, expected, pct))

    return percs, details


def arr_face_seconds(path, dt_start, dt_end, setting, stat_name):
    default_weight = dict_get_or_set(setting, "default_weight", 1.0)
    timeout_secs = dict_get_or_set(setting, "default_timeout", 60)

    # строим базовый список по секундам
    total_secs = int((dt_end - dt_start).total_seconds()) + 1
    times = [(dt_start + timedelta(seconds=i)).time()
                  for i in range(total_secs)]
    detections = [None] * total_secs
    weights_arr = [default_weight] * total_secs

    # если файла нет — сразу возвращаем все default_weight
    if not os.path.exists(path):
        return list(zip(times, detections, weights_arr))

    # достаём и сортируем метки по убыванию веса
    face_weights = dict_get_or_set(setting, "face_weights", {})
    # например: {"Name":1.0,"Unknown":0.9,"None":0.5}
    labels_by_weight = [lbl for lbl, _ in
                        sorted(face_weights.items(),
                               key=lambda kv: kv[1],
                               reverse=True)]

    # читаем CSV, заполняем detections
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                ts = datetime.strptime(f"{row[0]} {row[1]}",
                                       "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            if ts < dt_start or ts > dt_end:
                continue

            raw_labels = row[2:]
            # ищем самую «тяжёлую» метку в этой строке
            for lbl in labels_by_weight:
                if lbl == "Name" and stat_name in raw_labels:
                    chosen = "Name"
                    break
                if lbl == "Unknown" and any("Unknown" in r for r in raw_labels):
                    chosen = "Unknown"
                    break
                if lbl == "None" and any(r == "None" for r in raw_labels):
                    chosen = "None"
                    break
            else:
                chosen = "None"

            idx = int((ts.replace(microsecond=0) - dt_start).total_seconds())
            detections[idx] = chosen

    # заводим таймеры для каждой метки
    counters = {lbl: 0 for lbl in face_weights.keys()}

    # проходим по всем секундам
    for i in range(total_secs):
        det = detections[i]
        # если встретили детект — запускаем таймер для этой метки
        if det is not None:
            counters[det] = timeout_secs

        # выбираем максимальный вес из всех меток с таймером > 0
        active = [face_weights[lbl]
                  for lbl, ct in counters.items() if ct > 0]
        if active:
            w = max(active)
        else:
            w = default_weight

        weights_arr[i] = w

        # в конце цикла уменьшаем все таймеры
        for lbl in counters:
            if counters[lbl] > 0:
                counters[lbl] -= 1

    return list(zip(times, detections, weights_arr))

def arr_yolo_seconds(path, dt_start, dt_end, setting):
    # параметры
    default_weight = dict_get_or_set(setting, "default_weight", 1.0)
    timeout_secs = dict_get_or_set(setting, "default_timeout", 60)
    max_counter = timeout_secs * 2
    detect_threshold = int(timeout_secs * 1.1)
    # подготовка времени
    total_secs = int((dt_end - dt_start).total_seconds()) + 1
    times = [(dt_start + timedelta(seconds=i)).time() for i in range(total_secs)]

    # если файла нет — всё default_weight
    if not os.path.exists(path):
        return [(t, [], default_weight) for t in times]

    # веса предметов
    yolo_weights = dict_get_or_set(setting, "yolo_weights", {})
    # читаем события
    events_by_sec: dict[int, list[str]] = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                ts = datetime.strptime(f"{row[0]} {row[1]}", "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            if ts < dt_start or ts > dt_end:
                continue

            sec_idx = int((ts.replace(microsecond=0) - dt_start).total_seconds())
            raw_labels = row[2:]
            for lbl in yolo_weights:
                if lbl in raw_labels:
                    events_by_sec.setdefault(sec_idx, []).append(lbl)

    # заводим счётчики
    counters = {lbl: 0 for lbl in yolo_weights}
    # формируем результат
    result: list[tuple[datetime.time, list[str], float]] = []
    for i in range(total_secs):
        # обновляем счётчики по событиям этой секунды
        for lbl in events_by_sec.get(i, []):
            counters[lbl] = min(counters[lbl] + timeout_secs, max_counter)
        # определяем активные предметы
        active = [lbl for lbl, cnt in counters.items() if cnt >= detect_threshold]
        # вычисляем вес
        if active:
            w = 1.0
            for lbl in active:
                w *= yolo_weights.get(lbl, default_weight)
        else:
            w = default_weight
        # сохраняем
        result.append((times[i], active, w))
        # уменьшаем все счётчики на 1
        for lbl in counters:
            if counters[lbl] > 0:
                counters[lbl] -= 1

    return result

def arr_yamn_vosk_seconds(path_yamn, path_vosk, dt_start, dt_end, setting):
    default_weight = dict_get_or_set(setting, "default_weight", 1.0)
    audio_timeout = int(dict_get_or_set(setting, "audio_time_recognition", 0) * 1.5)
    yamnet_weights = dict_get_or_set(setting, "yamnet_weights", {})
    vosk_weights = dict_get_or_set(setting, "vosk_weights", {})

    # строим временную шкалу
    total_secs = int((dt_end - dt_start).total_seconds()) + 1
    times = [(dt_start + timedelta(seconds=i)).time() for i in range(total_secs)]

    # если оба файла отсутствуют — сразу всё default_weight
    if not os.path.exists(path_yamn) and not os.path.exists(path_vosk):
        return [(t, [], default_weight) for t in times]

    # парсим yamnet: sec_idx -> label
    events_yam: dict[int, str] = {}
    if os.path.exists(path_yamn):
        with open(path_yamn, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3: continue
                try:
                    ts = datetime.strptime(f"{row[0]} {row[1]}",
                                           "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
                if ts < dt_start or ts > dt_end:
                    continue
                idx = int((ts.replace(microsecond=0) - dt_start).total_seconds())
                events_yam[idx] = row[2]  # one of "None","speech","typing","media"

    # парсим vosk: sec_idx -> True, если есть слова
    events_vosk: dict[int, bool] = {}
    if os.path.exists(path_vosk):
        with open(path_vosk, newline='', encoding='utf-8', errors='ignore') as f: # Ща игнорирование
        # Или отрыть в виндовой кодировке with open(path_vosk, newline='', encoding='cp1251') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3: continue
                try:
                    ts = datetime.strptime(f"{row[0]} {row[1]}",
                                           "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
                if ts < dt_start or ts > dt_end:
                    continue
                idx = int((ts.replace(microsecond=0) - dt_start).total_seconds())
                # любое не-"None" в третьем столбце считаем «речь»
                events_vosk[idx] = row[2] != "None"

    # заводим счётчики для yamnet-меток (кроме "None")
    counters = {lbl: 0 for lbl in yamnet_weights if lbl != "None"}

    result: list[tuple[datetime.time, list[str], float]] = []
    for i in range(total_secs):
        # 1) обновляем счётчики по yamnet
        lbl_y = events_yam.get(i)
        if lbl_y and lbl_y in counters:
            counters[lbl_y] = audio_timeout
        # 2) если yamnet не дал «speech», но vosk отметил речь — ставим speech
        elif events_vosk.get(i, False):
            counters["speech"] = audio_timeout

        # 3) активные метки
        active = [lbl for lbl, ct in counters.items() if ct > 0]

        # 4) вычисляем вес
        if active:
            w = 1.0
            for lbl in active:
                w *= yamnet_weights.get(lbl, default_weight)
        else:
            w = default_weight

        result.append((times[i], active, w))

        # 5) уменьшаем все счётчики
        for lbl in counters:
            if counters[lbl] > 0:
                counters[lbl] -= 1

    return result


def get_stats(setting: dict, stat_name: str, stat_period: []):
    n = len(LOGS_FILES)
    if not (len(LOGS_OLDS) == n == len(TIME_KEYS)):
        raise ValueError("Ошибка stats.py: списки LOGS_* разной длины")

    date_obj, t0, t1 = stat_period
    dt_start = datetime.combine(date_obj, t0)
    dt_end = datetime.combine(date_obj, t1)
    if (dt_end - dt_start).total_seconds() <= 0:
        raise ValueError("Некорректный период: конец раньше начала")

    # Процент статистики
    percs, details = compute_log_percentages(setting, date_obj, dt_start, dt_end)
    percent_stats = sum(percs) / n if n else 0.0

    # print(f"Статистика по сотруднику {stat_name}:")
    # for fname, path, cnt, exp, pct in details:
    #     print(f"  {fname}: {cnt} / {exp} → {pct:.1f}% (файл {'есть' if os.path.exists(path) else 'нет'})")
    # print(f"Общий процент: {percent_stats:.1f}%")


    # Сегментация face
    path_face, _ = get_log_path(LOGS_FILES[0], LOGS_OLDS[0], date_obj)
    arr_face = arr_face_seconds(path_face, dt_start, dt_end, setting, stat_name)
    # print('face', *arr_face, sep='\n')

    # Сегментация yolo
    path_yolo, _ = get_log_path(LOGS_FILES[1], LOGS_OLDS[1], date_obj)
    arr_yolo = arr_yolo_seconds(path_yolo, dt_start, dt_end, setting)
    # print('yolo', *arr_yolo, sep='\n')

    # Сегментация yamnet и vosk
    path_vosk, _ = get_log_path(LOGS_FILES[2], LOGS_OLDS[2], date_obj)
    path_yamn, _ = get_log_path(LOGS_FILES[3], LOGS_OLDS[3], date_obj)
    arr_audio = arr_yamn_vosk_seconds(path_yamn, path_vosk, dt_start, dt_end, setting)
    # print('audio', *arr_audio, sep='\n')

    # Общий массив всех сегментов
    arr_merged = []
    for face, yolo, audio in zip(arr_face, arr_yolo, arr_audio):
        t_face, *_ , w_face = face
        t_yolo, *_ , w_yolo = yolo
        t_audio, *_, w_audio = audio
        # Усредняем веса
        mean_w = (w_face + w_yolo + w_audio) / 3
        # Сохраняем
        arr_merged.append((t_face, mean_w))
    # print('merged', *arr_merged, sep='\n')

    # Процент работы
    if arr_merged:
        percent_work = sum(w for _, w in arr_merged) / len(arr_merged)
    else:
        percent_work = 0.0

    return percent_stats, percent_work
