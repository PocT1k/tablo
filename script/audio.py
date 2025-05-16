import sounddevice as sd
import json
from vosk import Model, KaldiRecognizer
from datetime import datetime
import numpy as np
import time

from load import dict_get_or_set, archive_file
from conf import (VOSK_MODEL_PATH, YAMNET_MODEL_PATH, # модели
    VOSK_WORLD_PATH, VOSK_WORLD_OLD_DIR,  # файлы логирования
    YAMNET_INDICES_PATH, YAMNET_INDICES_OLD_DIR)


class AudioProcessor:
    def __init__(self, settings=None):
        self.settings = settings or {}
        # Индекс микрофона
        self.mic_index = dict_get_or_set(self.settings, "microphone", 0)
        # Интервал распознавания (секунды)
        self.recognition_time = dict_get_or_set(self.settings, "recognition_time", 5)

        # VOSK
        # Архивация лог-файла
        archive_file(VOSK_WORLD_PATH, VOSK_WORLD_OLD_DIR, True)
        # Загрузка модели Vosk (русская)
        try:
            self.vosk_model = Model(str(VOSK_MODEL_PATH))
            print("[VOSK] Модель Vosk загружена успешно.")
        except Exception as e:
            print("[VOSK ERROR] Не удалось загрузить модель Vosk:", e)
            raise
        self.recognizer = KaldiRecognizer(self.vosk_model, 16000)
        self.samplerate = 16000

        # YAMNet
        self.yamnet_sr = dict_get_or_set(self.settings, "yamnet_sr", 16000)
        self.recognition_time_yamnet = dict_get_or_set(self.settings, "recognition_time_yamnet", 3)
        self.yamnet_threshold = dict_get_or_set(self.settings, "indices_recognition_index", 0.6)
        self.detectors = dict_get_or_set(self.settings, "target_indices_yamnet", {})
        # Архивация логов YAMNet
        archive_file(YAMNET_INDICES_PATH, YAMNET_INDICES_OLD_DIR, True)
        # Загрузка модели YAMNet и меток

        # В try
        # import tensorflow_hub as hub
        # self.yamnet_model = hub.load(str(YAMNET_MODEL_PATH))
        # class_map = self.yamnet_model.class_map_path().numpy().decode()
        # with open(class_map, encoding="utf-8") as f:
        #     lines = f.read().splitlines()
        # # lines[0] — "index,mid,display_name"
        # self.class_names = [line.split(",", 2)[2] for line in lines[1:]]
        # print(f"[YAMNet] Loaded model, will classify every {self.recognition_time_yamnet}s, threshold={self.yamnet_threshold}")

        self.running_recognition_world = False
        self.running_classification_indices = False

    def start_microphone(self):
        idx = self.mic_index
        print(f"[AUDIO] Попытка открыть микрофон idx={idx}")
        try:
            self.stream = sd.InputStream(
                device=idx,
                channels=1,
                samplerate=self.samplerate
            )
            self.stream.start()
            print(f"[AUDIO] Микрофон {idx} успешно открыт")
            return True
        except Exception as e:
            print(f"[AUDIO ERROR] Не удалось открыть микрофон {idx}: {e}")
            self.last_error = str(e)
            return False

    def stop_processing(self):
        self.running_recognition_world = False
        self.running_classification_indices = False
        # stop_microphone
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def recognize_audio(self):
        # Ленивая загрузка YAMNet: при первом вызове
        if not hasattr(self, "_yamnet_ready"):
            try:
                import tensorflow_hub as hub
            except ImportError:
                print("[YAMNet] tensorflow_hub not available — skipping classification")
                return
            # загружаем модель и метки
            try:
                self.yamnet_model = hub.load(str(YAMNET_MODEL_PATH))
                class_map = self.yamnet_model.class_map_path().numpy().decode()
                with open(class_map, encoding="utf-8") as f:
                    lines = f.read().splitlines()
                self.class_names = [line.split(",", 2)[2] for line in lines[1:]]
                print(f"[YAMNet] Model loaded, will classify every {self.recognition_time_yamnet}s")
            except Exception as e:
                print("[YAMNet] Error loading model:", e)
                return
            self._yamnet_ready = True

        # распознавание речи каждые recognition_time секунд
        print(f"[AUDIO] Начало работы звукаобработки, каждые {self.recognition_time} секунд")
        while self.running_recognition_world:
            # print(f"[VOSK]Запись аудио на {self.recognition_time} секунд…")
            audio = sd.rec(
                int(self.recognition_time * self.samplerate),
                samplerate=self.samplerate,
                channels=1,
                dtype='int16'
            )
            sd.wait()

            # VOSK
            data = audio.tobytes()

            if self.recognizer.AcceptWaveform(data):
                res = json.loads(self.recognizer.Result())
                text = res.get("text", "").strip()
            else:
                res = json.loads(self.recognizer.PartialResult())
                text = res.get("partial", "").strip()

            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")

            with open(VOSK_WORLD_PATH, "a", encoding="cp1251", newline="") as f:
                if not text:
                    f.write(f"{date_str},{time_str},None\n")
                    print(f"[VOSK] {date_str} {time_str} - None")
                else:
                    safe_text = text.replace(",", " ") # экранируем запятые
                    f.write(f"{date_str},{time_str},{safe_text}\n")
                    print(f"[VOSK] {date_str} {time_str} - {text}")

            # YAMNet
            if self.yamnet_model:
                wav = np.squeeze(audio)

                scores, _, _ = self.yamnet_model(wav)
                mean_scores = np.mean(scores, axis=0)

                now = datetime.now()
                date_str = now.strftime("%Y-%m-%d")
                time_str = now.strftime("%H:%M:%S")

                detected = []
                for indices in self.detectors.values():
                    if not indices:
                        continue
                    best_idx = max(indices, key=lambda i: mean_scores[i])
                    if mean_scores[best_idx] >= self.yamnet_threshold:
                        detected.append(self.class_names[best_idx])
                if not detected:
                    detected = ["None"]

                with open(YAMNET_INDICES_PATH, "a", encoding="utf-8", newline="") as f:
                    f.write(f"{date_str},{time_str},{','.join(detected)}\n")
                print(f"[YAMNet] {date_str} {time_str} - {', '.join(detected)}")

    # YAMNet
    def classify_indices(self):
        from datetime import datetime

        # Ленивая загрузка YAMNet: при первом вызове
        if not hasattr(self, "_yamnet_ready"):
            try:
                import tensorflow_hub as hub
            except ImportError:
                print("[YAMNet] tensorflow_hub not available — skipping classification")
                return
            # загружаем модель и метки
            try:
                self.yamnet_model = hub.load(str(YAMNET_MODEL_PATH))
                class_map = self.yamnet_model.class_map_path().numpy().decode()
                with open(class_map, encoding="utf-8") as f:
                    lines = f.read().splitlines()
                self.class_names = [line.split(",", 2)[2] for line in lines[1:]]
                print(f"[YAMNet] Model loaded, will classify every {self.recognition_time_yamnet}s")
            except Exception as e:
                print("[YAMNet] Error loading model:", e)
                return
            self._yamnet_ready = True

        # если мы сюда попали, то YAMNet запустился
        while self.running_classification_indices:
            audio = sd.rec(
                int(self.recognition_time_yamnet * self.yamnet_sr),
                samplerate=self.yamnet_sr,
                channels=1,
                dtype="float32"
            )
            sd.wait()
            wav = np.squeeze(audio)

            scores, _, _ = self.yamnet_model(wav)
            mean_scores = np.mean(scores, axis=0)

            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")

            detected = []
            for indices in self.detectors.values():
                if not indices:
                    continue
                best_idx = max(indices, key=lambda i: mean_scores[i])
                if mean_scores[best_idx] >= self.yamnet_threshold:
                    detected.append(self.class_names[best_idx])
            if not detected:
                detected = ["None"]

            with open(YAMNET_INDICES_PATH, "a", encoding="utf-8", newline="") as f:
                f.write(f"{date_str},{time_str},{','.join(detected)}\n")
            print(f"[YAMNet] {date_str} {time_str} - {', '.join(detected)}")
        time.sleep(self.recognition_time_yamnet)

    # def classify_indices(self):
    #     from datetime import datetime
    #
    #     while self.running_classification_indices:
    #         audio = sd.rec(
    #             int(self.recognition_time_yamnet * self.yamnet_sr),
    #             samplerate=self.yamnet_sr,
    #             channels=1,
    #             dtype="float32"
    #         )
    #         sd.wait()
    #         wav = np.squeeze(audio)
    #
    #         scores, _, _ = self.yamnet_model(wav)
    #         mean_scores = np.mean(scores, axis=0)
    #
    #         now = datetime.now()
    #         date_str = now.strftime("%Y-%m-%d")
    #         time_str = now.strftime("%H:%M:%S")
    #
    #         # Собираем найденные звуки по всем категориям
    #         detected = []
    #         for indices in self.detectors.values():
    #             if not indices:
    #                 continue
    #             best_idx = max(indices, key=lambda i: mean_scores[i])
    #             if mean_scores[best_idx] >= self.yamnet_threshold:
    #                 detected.append(self.class_names[best_idx])
    #
    #         if not detected:
    #             detected = ["None"]
    #
    #         # Лог в файл
    #         with open(YAMNET_INDICES_PATH, "a", encoding="utf-8", newline="") as f:
    #             f.write(f"{date_str},{time_str},{','.join(detected)}\n")
    #         # И в консоль
    #         print(f"[YAMNet] {date_str} {time_str} - {', '.join(detected)}")
