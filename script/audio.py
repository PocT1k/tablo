import sounddevice as sd
import json
from vosk import Model, KaldiRecognizer
from datetime import datetime
import numpy as np

from load import dict_get_or_set, archive_file_by_date, write_attendance_dated
from conf import (VOSK_MODEL_PATH, YAMNET_MODEL_PATH, # модели
    VOSK_WORLD_PATH, VOSK_WORLD_OLD_DIR,  # файлы логирования
    YAMNET_INDICES_PATH, YAMNET_INDICES_OLD_DIR)


class AudioProcessor:
    def __init__(self, settings=None):
        self.settings = settings or {}
        self.recognition_time = dict_get_or_set(self.settings, "recognition_time", 5)
        self.recognition_time_add = dict_get_or_set(self.settings, "recognition_time_add", 0.5)
        print(f"[AUDIO] Окно = {self.recognition_time}с.+{self.recognition_time_add}с.")

        # VOSK
        self.mic_index = dict_get_or_set(self.settings, "microphone", 0)
        self.samplerate = dict_get_or_set(self.settings, "vosk_sr", 16000)
        # Архивация VOSK-логов
        archive_file_by_date(VOSK_WORLD_PATH, VOSK_WORLD_OLD_DIR, True)
        # Попытка загрузить Vosk
        self.vosk_ok = False
        try:
            self.vosk_model = Model(str(VOSK_MODEL_PATH))
            self.recognizer = KaldiRecognizer(self.vosk_model, self.samplerate)
            self.vosk_ok = True
            print("[VOSK] Модель Vosk загружена успешно.")
        except Exception as e:
            print("[VOSK ERROR] Не удалось загрузить Vosk:", e)

        # YAMNet
        self.yamnet_sr = dict_get_or_set(self.settings, "yamnet_sr", 16000)
        self.yamnet_threshold = dict_get_or_set(self.settings, "indices_recognition_index", 0.6)
        self.detectors = dict_get_or_set(self.settings, "target_indices_yamnet", {})
        # Архивация YAMNet-логов
        archive_file_by_date(YAMNET_INDICES_PATH, YAMNET_INDICES_OLD_DIR, True)
        # Попытка загрузить YAMNet
        self.yamnet_ok = False
        try:
            import tensorflow_hub as hub
            self.yamnet_model = hub.load(str(YAMNET_MODEL_PATH))
            class_map = self.yamnet_model.class_map_path().numpy().decode()
            with open(class_map, encoding="utf-8") as f:
                lines = f.read().splitlines()
            self.class_names = [line.split(",",2)[2] for line in lines[1:]]
            self.yamnet_ok = True
            print(f"[YAMNet] Модель загружена, порог={self.yamnet_threshold}")
        except Exception as e:
            print("[YAMNet ERROR] Не удалось загрузить YAMNet:", e)

        # Флаги запуска циклов
        self.running_recognition = False

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

    def stop_microphone(self):
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def stop_processing(self):
        self.running_recognition = False
        self.stop_microphone()

    def recognize_vosk(self, audio_bytes, timestamp):
        if not self.vosk_ok:
            return

        self.recognizer.Reset() # Сброс состояния - буффера
        if self.recognizer.AcceptWaveform(audio_bytes):
            res = json.loads(self.recognizer.Result())
            text = res.get("text", "").strip()
        else:
            res = json.loads(self.recognizer.PartialResult())
            text = res.get("partial", "").strip()

        # логирование
        write_attendance_dated(VOSK_WORLD_PATH, text, timestamp, '[VOSK]')

    def recognize_yamnet(self, audio_array, timestamp):
        if not self.yamnet_ok:
            return
        wav = np.squeeze(audio_array)
        scores, _, _ = self.yamnet_model(wav)
        mean_scores = np.mean(scores, axis=0)

        detected = []
        for indices in self.detectors.values():
            if not indices:
                continue
            best = max(indices, key=lambda i: mean_scores[i])
            if mean_scores[best] >= self.yamnet_threshold:
                detected.append(self.class_names[best])
        if not detected:
            detected = ["None"]

        line = ",".join(detected)
        write_attendance_dated(YAMNET_INDICES_PATH, line, timestamp, '[YAMNet]')

    def recognize_audio(self):
        self.running_recognition = True
        tail_len = int(self.samplerate * self.recognition_time_add)
        prev_tail = None

        while self.running_recognition:
            # Запись recognition_time секунд
            audio = sd.rec(
                int(self.recognition_time * self.samplerate),
                samplerate=self.samplerate,
                channels=1,
                dtype='int16'
            )
            sd.wait()
            now = datetime.now().strftime("%Y-%m-%d,%H:%M:%S")

            # Собираем buffer = хвост прошлой + свежий кусок
            if prev_tail is None:
                buffer = audio
            else:
                buffer = np.concatenate([prev_tail, audio], axis=0)

            # Готовим prev_tail для следующей итерации
            prev_tail = buffer[-tail_len:].copy() if buffer.shape[0] > tail_len else buffer.copy()

            # Vosk
            if self.vosk_ok:
                audio_bytes = buffer.tobytes()
                self.recognize_vosk(audio_bytes, now)

            # YAMNet
            if self.yamnet_ok:
                buf_f32 = buffer.astype('float32') / np.iinfo('int16').max
                self.recognize_yamnet(buf_f32, now)
