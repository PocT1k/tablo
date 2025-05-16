import sounddevice as sd
import json
from vosk import Model, KaldiRecognizer
from pathlib import Path
from datetime import datetime

from load import dict_get_or_set, archive_file
from conf import vosk_model_path # модели
from conf import vosk_world_path, vosk_world_old_path # пути логирования


class AudioProcessor:
    def __init__(self, settings=None):
        self.settings = settings or {}
        # Индекс микрофона
        self.mic_index = dict_get_or_set(self.settings, "microphone", 0)
        # Интервал распознавания (секунды)
        self.recognition_time = dict_get_or_set(self.settings, "recognition_time_world", 5)
        # Архивация лог-файла
        archive_file(vosk_world_path, vosk_world_old_path, True)
        # Загрузка модели Vosk (русская)
        try:
            self.vosk_model = Model(str(vosk_model_path))
            print("[AUDIO] Модель Vosk загружена успешно.")
        except Exception as e:
            print("[AUDIO ERROR] Не удалось загрузить модель Vosk:", e)
            raise

        self.recognizer = KaldiRecognizer(self.vosk_model, 16000)
        self.samplerate = 16000

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

    def recognize_loop(self):
        # распознавание речи каждые recognition_time секунд

        print(f"Начало работы ASR. Каждые {self.recognition_time} секунд выводится транскрипт RU.")
        while True:
            print(f"Запись аудио на {self.recognition_time} секунд…")
            audio = sd.rec(
                int(self.recognition_time * self.samplerate),
                samplerate=self.samplerate,
                channels=1,
                dtype='int16'
            )
            sd.wait()
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

            with open(vosk_world_path, "a", encoding="cp1251", newline="") as f:
                if not text:
                    f.write(f"{date_str},{time_str},None\n")
                    print(f"[RECOGNITION] {date_str} {time_str} - None")
                else:
                    safe_text = text.replace(",", " ") # экранируем запятые
                    f.write(f"{date_str},{time_str},{safe_text}\n")
                    print(f"[RECOGNITION] {date_str} {time_str} - {text}")
