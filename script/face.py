# face.py
import cv2
import numpy as np
import face_recognition
from datetime import datetime
from funk import dict_get_or_set


class FaceProcessor:
    def __init__(self, model, settings=None):
        """
        :param model: обученная модель для распознавания лиц (SVC)
        :param settings: полный JSON со всеми ключами, включая 'camera' и 'recognition_time'
        """
        self.model = model
        self.settings = settings or {}

        # Интервал распознавания (секунды) и индекс камеры
        self.recognition_time = dict_get_or_set(self.settings, "recognition_time", 1)
        self.camera_index = dict_get_or_set(self.settings, "camera", 0)

        self.video_capture = None
        # Для кэширования результатов между распознаваниями
        self.last_detection_time = None
        self.last_results = []

        # Для отметки посещаемости
        self.last_attendance = {}

    def start_camera(self):
        """Запуск видеопотока по индексу из настроек"""
        self.video_capture = cv2.VideoCapture(self.camera_index)
        return self.video_capture.isOpened()

    def stop_camera(self):
        """Остановка видеопотока"""
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None

    def get_frame(self):
        if self.video_capture and self.video_capture.isOpened():
            return self.video_capture.read()
        return False, None

    def process_frame(self, frame):
        """
        Если с последнего распознавания прошло >= recognition_time,
        выполняем детекцию + распознавание и обновляем last_results,
        иначе просто рисуем предыдущие рамки.
        Также при распознавании отмечаем attendance.csv.
        """
        now = datetime.now()
        do_detect = (
            self.last_detection_time is None or
            (now - self.last_detection_time).total_seconds() >= self.recognition_time
        )

        processed = frame.copy()

        if do_detect:
            # обновляем время и очищаем прошлые результаты
            self.last_detection_time = now
            self.last_results = []

            # масштаб для детекции
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locs = face_recognition.face_locations(rgb_small)
            encs = face_recognition.face_encodings(rgb_small, locs)

            for (top, right, bottom, left), enc in zip(locs, encs):
                probs = self.model.predict_proba([enc])[0]
                idx = np.argmax(probs)
                name = self.model.classes_[idx]
                conf = probs[idx]
                if conf < 0.6:
                    name = "Unknown"

                # масштаб обратно
                top *= 2; right *= 2; bottom *= 2; left *= 2

                # рисуем рамку
                cv2.rectangle(processed, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(
                    processed,
                    f"{name} ({conf*100:.1f}%)",
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
                )

                # отметка посещаемости
                if name != "Unknown":
                    last_t = self.last_attendance.get(name)
                    if last_t is None or (now - last_t).total_seconds() >= self.recognition_time:
                        ts = now.strftime("%Y-%m-%d %H:%M:%S")
                        with open("attendance.csv", "a") as f:
                            f.write(f"{name},{ts}\n")
                        print(f"[ATTENDANCE] {name} отмечен(а) в {ts}")
                        self.last_attendance[name] = now

                self.last_results.append({
                    "name": name,
                    "confidence": float(conf),
                    "location": (top, right, bottom, left)
                })
        else:
            # просто рисуем прошлые рамки
            for res in self.last_results:
                top, right, bottom, left = res["location"]
                name = res["name"]
                conf = res["confidence"]
                cv2.rectangle(processed, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(
                    processed,
                    f"{name} ({conf*100:.1f}%)",
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
                )

        return processed, self.last_results
