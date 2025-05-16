# audio.py
import sounddevice as sd
from load import dict_get_or_set

class AudioProcessor:
    def __init__(self, settings=None):
        self.settings = settings or {}
        # По умолчанию микрофон = 0
        self.mic_index = dict_get_or_set(self.settings, "microphone", 0)
        self.stream = None

    def start_microphone(self):
        # Debug: список всех устройств и выбранный индекс
        devs = sd.query_devices()
        inputs = [d for d in devs if d['max_input_channels'] > 0]
        print("[AUDIO] Все устройства ввода:")
        for i, d in enumerate(inputs):
            print(f"  [{i}] index={d['index']} name={d['name']}")

        idx = self.mic_index
        print(f"[AUDIO] Попытка открыть микрофон idx={idx}")

        try:
            self.stream = sd.InputStream(
                device=idx,
                channels=1,
                samplerate=44100,
                callback=self._callback
            )
            self.stream.start()
            print(f"[AUDIO] Микрофон {idx} успешно открыт")
            return True
        except Exception as e:
            # Логируем и возвращаем False
            print(f"[AUDIO ERROR] Не удалось открыть микрофон {idx}: {e}")
            # Сохраним текст ошибки, чтобы UI мог его показать
            self.last_error = str(e)
            return False

    def stop_microphone(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _callback(self, indata, frames, time, status):
        # ваша обработка аудио
        pass
