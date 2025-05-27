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


def arr_face_seconds(path, dt_start, dt_end, setting, stat_name, found_things):
    default_weight = dict_get_or_set(setting, "default_weight", 1.0)
    timeout_secs = dict_get_or_set(setting, "default_timeout", 60)
    found_things['NotFace'] = 0

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

        if det == "None":
            found_things['NotFace'] += 1

        # в конце цикла уменьшаем все таймеры
        for lbl in counters:
            if counters[lbl] > 0:
                counters[lbl] -= 1

    return list(zip(times, detections, weights_arr))

def arr_yolo_seconds(path, dt_start, dt_end, setting, found_things):
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
    # инициализируем счётчики у found_things
    for lbl in yolo_weights:
        if lbl != "None":
            found_things.setdefault(lbl, 0)

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

        for lbl in active:
            if lbl != "None":
                found_things[lbl] += 1

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

def arr_yamn_vosk_seconds(path_yamn, path_vosk, dt_start, dt_end, setting, found_things):
    # Загрузка параметров
    default_weight = dict_get_or_set(setting, "default_weight", 1.0)
    audio_timeout = int(dict_get_or_set(setting, "audio_time_recognition", 0) * 1.5)
    yamnet_weights = dict_get_or_set(setting, "yamnet_weights", {})
    vosk_weights = dict_get_or_set(setting, "vosk_weights", {})

    # Инициализация счётчика найденных событий
    # для всех Yamnet-меток (кроме "None") и для "speech" из Vosk
    for lbl in yamnet_weights:
        if lbl != "None":
            found_things.setdefault(lbl, 0)
    if "speech" in vosk_weights:
        found_things.setdefault("speech", 0)

    # Построение временной шкалы
    total_secs = int((dt_end - dt_start).total_seconds()) + 1
    times = [(dt_start + timedelta(seconds=i)).time() for i in range(total_secs)]

    # Если нет ни одного файла — сразу возвращаем всё default_weight
    if not os.path.exists(path_yamn) and not os.path.exists(path_vosk):
        return [(t, [], default_weight) for t in times]

    # Чтение Vosk (любая запись ≠ "None" считается речью)
    events_vosk = {}
    if os.path.exists(path_vosk):
        with open(path_vosk, newline='', encoding='utf-8', errors='ignore') as f:
            for row in csv.reader(f):
                if len(row) < 3: continue
                try:
                    ts = datetime.strptime(f"{row[0]} {row[1]}", "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
                if dt_start <= ts <= dt_end and row[2] != "None":
                    idx = int((ts.replace(microsecond=0) - dt_start).total_seconds())
                    events_vosk[idx] = True

    # Чтение Yamnet (берём ровно одну метку в третьем столбце)
    events_yam = {}
    if os.path.exists(path_yamn):
        with open(path_yamn, newline='', encoding='utf-8') as f:
            for row in csv.reader(f):
                if len(row) < 3: continue
                try:
                    ts = datetime.strptime(f"{row[0]} {row[1]}", "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
                if dt_start <= ts <= dt_end:
                    idx = int((ts.replace(microsecond=0) - dt_start).total_seconds())
                    events_yam[idx] = row[2]  # "None", "speech", "typing", "media", …

    # Словарь-таймеры для каждой метки
    counters = {lbl: 0 for lbl in yamnet_weights if lbl != "None"}
    counters["speech"] = 0

    # Основной проход по каждой секунде
    result = []
    for i in range(total_secs):
        # 8.1) Vosk-процессинг (первым)
        if events_vosk.get(i, False):
            counters["speech"] = audio_timeout

        # Yamnet-процессинг (вторым, может перезаписать)
        lbl_y = events_yam.get(i)
        if lbl_y and lbl_y != "None":
            counters[lbl_y] = audio_timeout

        # Составляем список активных меток
        active = [lbl for lbl, ct in counters.items() if ct > 0]

        # Выбираем единственную «главную» метку с наибольшим весом
        if not active:
            chosen = None
            w = default_weight
        else:
            chosen, w = max(
                ((lbl,
                  vosk_weights.get(lbl, default_weight)
                  if lbl == "speech"
                  else yamnet_weights.get(lbl, default_weight))
                 for lbl in active),
                key=lambda x: x[1]
            )
            # Инкрементируем счётчик в found_things
            found_things[chosen] += 1

        # Записываем результат: возвращаем либо [] (если None), либо [chosen]
        result.append((times[i], [] if chosen is None else [chosen], w))

        # Уменьшаем все таймеры на 1
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

    #Массив найденных предметов
    found_things = {}

    # Сегментация face
    path_face, _ = get_log_path(LOGS_FILES[0], LOGS_OLDS[0], date_obj)
    arr_face = arr_face_seconds(path_face, dt_start, dt_end, setting, stat_name, found_things)
    # print('face', *arr_face, sep='\n')

    # Сегментация yolo
    path_yolo, _ = get_log_path(LOGS_FILES[1], LOGS_OLDS[1], date_obj)
    arr_yolo = arr_yolo_seconds(path_yolo, dt_start, dt_end, setting, found_things)
    # print('yolo', *arr_yolo, sep='\n')

    # Сегментация yamnet и vosk
    path_vosk, _ = get_log_path(LOGS_FILES[2], LOGS_OLDS[2], date_obj)
    path_yamn, _ = get_log_path(LOGS_FILES[3], LOGS_OLDS[3], date_obj)
    arr_audio = arr_yamn_vosk_seconds(path_yamn, path_vosk, dt_start, dt_end, setting, found_things)
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

    found_things['len'] = len(arr_merged)
    return percent_stats, percent_work, found_things
