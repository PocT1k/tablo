import sounddevice as sd
# import tensorflow_hub as hub
import numpy as np
import json
from datetime import datetime

from conf import YAMNET_MODEL_PATH, YAMNET_INDICES_PATH, SETTING_JSON_PATH
from load import dict_get_or_set


# Загрузка настроек из JSON
with open(SETTING_JSON_PATH, "r", encoding="utf-8") as f:
    settings = json.load(f)

# Параметры из настроек
SR = dict_get_or_set(settings, "yamnet_sr", 16000)
WINDOW_SEC = dict_get_or_set(settings, "recognition_time_yamnet", 3)
THRESHOLD = dict_get_or_set(settings, "indices_recognition_index", 0.6)
detectors = dict_get_or_set(settings, "target_indices_yamnet", {})  # словарь категорий -> список индексов

# 1) Загрузка YAMNet из локальной папки
yam_model = hub.load(str(YAMNET_MODEL_PATH))

# 2) Чтение карты классов
class_map_path = yam_model.class_map_path().numpy().decode()
with open(class_map_path, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()
class_names = [line.split(",", 2)[2] for line in lines[1:]]  # индекс соответствует позиции +1 в lines

print(f"Streaming YAMNet classification every {WINDOW_SEC} sec. Threshold = {THRESHOLD}")

# 3) Основной цикл классификации
while True:
    print(f"[YAMNet]Recording audio for {WINDOW_SEC} seconds…")
    audio = sd.rec(int(SR * WINDOW_SEC), samplerate=SR, channels=1, dtype="float32")
    sd.wait()

    wav = np.squeeze(audio)
    scores, embeddings, spectrogram = yam_model(wav)
    mean_scores = np.mean(scores, axis=0)

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Сбор обнаруженных звуков по всем категориям
    detected = []
    for indices in detectors.values():
        if not indices:
            continue
        best_idx = max(indices, key=lambda i: mean_scores[i])
        if mean_scores[best_idx] >= THRESHOLD:
            detected.append(class_names[best_idx])

    # Если ничего не обнаружено — None
    if not detected:
        detected = ["None"]

    # Запись в файл одним строковым списком
    with open(YAMNET_INDICES_PATH, "a", encoding="utf-8", newline="") as f:
        f.write(f"{date_str},{time_str},{','.join(detected)}\n")

    # Вывод в консоль
    print(f"[YAMNet] {date_str} {time_str} - {', '.join(detected)}")
