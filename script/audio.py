import sounddevice as sd
import json
import numpy as np
import math
from datetime import datetime
from PyQt5.QtWidgets import QMessageBox
from vosk import Model, KaldiRecognizer
from collections import deque

from load import dict_get_or_set, archive_file_by_date, write_attendance_dated
from conf import (VOSK_MODEL_PATH, YAMNET_MODEL_PATH, # модели
    VOSK_WORLD_PATH, VOSK_WORLD_OLD_DIR,  # файлы логирования
    YAMNET_INDICES_PATH, YAMNET_INDICES_OLD_DIR)


class AudioProcessor:
    def __init__(self, settings=None):
        self.settings = settings or {}
        self.audio_detect_threshold = dict_get_or_set( self.settings, "audio_detect_threshold", 0.0005)
        self.audio_active_timeout = dict_get_or_set(self.settings, "audio_active_timeout", 1.0)

        self.recognition_time = dict_get_or_set(self.settings, "audio_time_recognition", 3)
        self.recognition_time_add = dict_get_or_set(self.settings, "audio_time_add_recognition", 0.5)
        self.mic_index = dict_get_or_set(self.settings, "microphone", 0)
        print(f"[AUDIO] Окно распознования звука = {self.recognition_time}с.+{self.recognition_time_add}с.")

        self.chunk_duration = 0.5
        self.chunks_per_recognition = math.ceil(self.recognition_time / self.chunk_duration)
        self.chunks_per_recognition_add = math.ceil(self.recognition_time_add / self.chunk_duration)
        self._chunk_buffer = deque(maxlen=self.chunks_per_recognition + self.chunks_per_recognition_add)
        self._chunk_counter = 0

        # Для работы детекта микрофона
        self._audio_active_counter = 0
        self.audio_active = False
        # Флаги запуска циклов
        self.running_recognition = False
        self.running_recognition_thread = False

        # VOSK
        self.samplerate = dict_get_or_set(self.settings, "vosk_sr", 16000)
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
            QMessageBox.warning(None, "Ошибка загрузки",
                                f'Модуль распознования слов VOSK отключён \nМодель не найдена \n{e}')

        # YAMNet
        self.yamnet_sr = dict_get_or_set(self.settings, "yamnet_sr", 16000)
        self.yamnet_threshold = dict_get_or_set(self.settings, "yamnet_threshold", 0.6)
        self.yamnet_indices = dict_get_or_set(self.settings, "yamnet_indices", {})
        self.yamnet_groups = {
            group: np.array(indices, dtype=int)
            for group, indices in self.yamnet_indices.items()
            if indices  # пропускаем пустые
        }
        archive_file_by_date(YAMNET_INDICES_PATH, YAMNET_INDICES_OLD_DIR, True)
        # Попытка загрузить YAMNet
        self.yamnet_ok = False
        try:
            if not self.yamnet_indices:
                raise ValueError("yamnet_indices не заданы в настройках")
            import tensorflow_hub as hub
            self.yamnet_model = hub.load(str(YAMNET_MODEL_PATH))
            class_map = self.yamnet_model.class_map_path().numpy().decode()
            with open(class_map, encoding="utf-8") as f:
                lines = f.read().splitlines()
            self.class_names = [line.split(",",2)[2] for line in lines[1:]]
            self.yamnet_ok = True
            print(f"[YAMN] Модель YAMNet загружена, порог распознования = {self.yamnet_threshold}")
        except Exception as e:
            print("[YAMN ERROR] Не удалось загрузить YAMNet:", e)
            QMessageBox.warning(None, "Ошибка загрузки",
                                f'Модуль распознования звуков YAMNET отключён \nМодель не найдена или ошибка подключения tensorflow_hub \n{e}')

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
            raise ValueError("Не удалось открыть микрофон")

    def stop_microphone(self):
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def stop_processing(self):
        self.running_recognition = False
        self.stop_microphone()

    def proc_vosk(self, audio_bytes, timestamp):
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

    def proc_yamnet(self, audio_array, timestamp):
        if not self.yamnet_ok:
            return

        wav = np.squeeze(audio_array)
        scores, _, _ = self.yamnet_model(wav)
        mean_scores = np.mean(scores, axis=0)

        detected = []

        for group, idx_arr in self.yamnet_groups.items():
            group_scores = mean_scores[idx_arr]
            best_local = idx_arr[np.argmax(group_scores)]
            if mean_scores[best_local] >= self.yamnet_threshold:
                detected.append(group)

        if not detected:
            detected = ["None"]
        write_attendance_dated(YAMNET_INDICES_PATH, detected, timestamp, '[YAMN]')

    def proc_audio(self):
        self.running_recognition = True
        self._chunk_counter = 0
        self._chunk_buffer.clear()

        while self.running_recognition:
            # записываем один «чанк» длительностью chunk_duration
            samples = int(self.chunk_duration * self.samplerate)
            audio_chunk = sd.rec(
                samples,
                samplerate=self.samplerate,
                channels=1,
                dtype='int16'
            )
            sd.wait()
            # В буффер добавляем, а старый сам стирается
            self._chunk_buffer.append(audio_chunk)

            # детект аудио-активности по RMS
            norm = audio_chunk.astype(np.float32) / np.iinfo('int16').max
            rms = np.sqrt(np.mean(norm ** 2))
            if rms > self.audio_detect_threshold:
                self.audio_active = True
                self._audio_active_counter = int(self.audio_active_timeout / self.recognition_time)
            else:
                if self._audio_active_counter > 0:
                    self._audio_active_counter -= 1
                else:
                    self.audio_active = False

            now = datetime.now().strftime("%Y-%m-%d,%H:%M:%S")
            # Обработка накопившегося буффера
            self._chunk_counter += 1
            if self._chunk_counter >= self.chunks_per_recognition:
                self._chunk_counter = 0
                audio_full = np.concatenate(self._chunk_buffer, axis=0)

                # Vosk
                if self.vosk_ok:
                    self.proc_vosk(audio_full.tobytes(), now)

                # YAMNet
                if self.yamnet_ok:
                    buf_f32 = audio_full.astype('float32') / np.iinfo('int16').max
                    self.proc_yamnet(buf_f32, now)

        self.running_recognition_thread = False