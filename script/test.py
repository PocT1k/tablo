import sounddevice as sd
import tensorflow_hub as hub
import numpy as np
from conf import yamnet_model_path

# 1) Загрузка YAMNet из локальной папки
yam_model = hub.load(str(yamnet_model_path))

# 2) Чтение карты классов, пропуская первую строку-заголовок
class_map_path = yam_model.class_map_path().numpy().decode()
with open(class_map_path, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()
# lines[0] == "index,mid,display_name"
class_names = lines[1:]

# 3) Параметры
SR = 16000
WINDOW_SEC = 3.0
THRESHOLD = 0.6  # порог детектирования

print(f"Streaming YAMNet classification every {WINDOW_SEC} sec. Threshold = {THRESHOLD}")

# 4) Потоковое распознавание
while True:
    print(f"\nRecording audio for {WINDOW_SEC} seconds…")
    audio = sd.rec(int(SR * WINDOW_SEC), samplerate=SR, channels=1, dtype="float32")
    sd.wait()  # ждём окончания записи

    # YAMNet ожидает одномерный массив float32
    wav = np.squeeze(audio)
    scores, embeddings, spectrogram = yam_model(wav)
    mean_scores = np.mean(scores, axis=0)

    # 5) Отбираем классы с уровнем ≥ THRESHOLD
    detections = [(i, mean_scores[i]) for i in range(1, len(mean_scores))  # type: ignore
                  if mean_scores[i] >= THRESHOLD]
    if detections:
        print("Detected sounds (score ≥ {:.2f}):".format(THRESHOLD))
        for idx, score in detections:
            # класс с индексом idx соответствует строке class_names[idx-1]
            label = class_names[idx - 1]
            print(f"  {idx},{label} — {score:.3f}")
    else:
        print("No sounds detected above threshold.")
