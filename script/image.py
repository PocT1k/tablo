import os
import pickle
import cv2
import numpy as np
from sklearn.svm import SVC
import face_recognition
from datetime import datetime

from load import dict_get_or_set, archive_file, archive_file_by_date, write_attendance_dated
from conf import (FACE_MODEL_PATH, # модели
    FACE_ATTENDANCE_PATH, FACE_ATTENDANCE_OLD_DIR, # файлы логирования
    DATASET_CONVERTED_DIR, DATASET_RAW_DIR) # датасеты


class ImageProcessor:
    def __init__(self, settings=None):
        self.settings = settings or {}
        # Интервал распознавания (секунды) и индекс камеры
        self.recognition_time = dict_get_or_set(self.settings, "recognition_time_image", 2)
        self.camera_index = dict_get_or_set(self.settings, "camera", 0)
        self.face_recognition_index = dict_get_or_set(self.settings, "recognition_index_face", 0.6)
        print(f"[VIDEO] Окно распознования изображения = {self.recognition_time}с.")
        print(f"[FACE] Порог распознования = {self.face_recognition_index}")

        self.video_capture = None
        # Для кэширования результатов между распознаваниями
        self.last_detection_time = None
        self.last_results = []
        # Для отметки посещаемости
        self.last_attendance = {}
        # Архивация attendance
        archive_file_by_date(FACE_ATTENDANCE_PATH, FACE_ATTENDANCE_OLD_DIR, True)

    def start_camera(self):
        if os.path.exists(FACE_MODEL_PATH):
            with open(FACE_MODEL_PATH, "rb") as f:
                self.face_model = pickle.load(f)
        else:
            self.face_model = None

        if self.face_model is None:
            print("[VIDEO] Модель не найдена, камеру не открываем")
            return False

        # попытка открыть выбранную камеру
        idx = self.camera_index
        print(f"[VIDEO] Попытка открыть камеру idx={idx}")
        self.video_capture = cv2.VideoCapture(idx)
        if not self.video_capture.isOpened():
            print(f"[VIDEO ERROR] Не удалось открыть камеру {idx}")
            return False

        print(f"[VIDEO] Камера {idx} успешно открыта")
        return True

    def stop_camera(self):
        """Остановка видеопотока"""
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None


    def get_frame(self):
        if self.video_capture and self.video_capture.isOpened():
            return self.video_capture.read()
        return False, None

    def proc_image(self, frame):
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

            # масштабируем для детекции
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locs = face_recognition.face_locations(rgb_small)
            encs = face_recognition.face_encodings(rgb_small, locs)

            # если лица не найдены — записываем None
            if not locs:
                write_attendance_dated(FACE_ATTENDANCE_PATH, 'None', None, '[FACE]')
            else:
                # обрабатываем каждое найденное лицо
                for (top, right, bottom, left), enc in zip(locs, encs):
                    probs = self.face_model.predict_proba([enc])[0]
                    idx = np.argmax(probs)
                    name = self.face_model.classes_[idx]
                    conf = probs[idx]
                    if conf < self.face_recognition_index:
                        name = "Unknown"

                    # масштаб обратно
                    top *= 2; right *= 2; bottom *= 2; left *= 2

                    # рисуем рамку и подпись
                    cv2.rectangle(processed, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(
                        processed,
                        f"{name} ({conf * 100:.1f}%)",
                        (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
                    )

                    # отметка посещаемости (для любого лица)
                    write_attendance_dated(FACE_ATTENDANCE_PATH, name, None, '[FACE]')
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
                    f"{name} ({conf * 100:.1f}%)",
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
                )

        return processed, self.last_results

    def _convert_image(self):
        os.makedirs(DATASET_CONVERTED_DIR, exist_ok=True)

        for student_name in os.listdir(DATASET_RAW_DIR):
            student_dir = os.path.join(DATASET_RAW_DIR, student_name)
            output_dir = os.path.join(DATASET_CONVERTED_DIR, student_name)
            os.makedirs(output_dir, exist_ok=True)

            if not os.path.isdir(student_dir):
                continue

            for img_name in os.listdir(student_dir):
                img_path = os.path.join(student_dir, img_name)

                # Пропускаем не-изображения
                valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
                if not img_name.lower().endswith(valid_extensions):
                    print(f"[SKIP IMG] Пропущен не-изображение: {img_path}")
                    continue

                image = cv2.imread(img_path)
                if image is None:
                    print(f"[WARNING IMG] Не удалось загрузить {img_path}")
                    continue

                try:
                    # image = cv2.resize(image, (256, 256)) # ресайз

                    # Улучшение контраста
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    gray_norm = clahe.apply(gray)

                    # После улучшения контраста GRAY → RGB
                    rgb_image = cv2.cvtColor(gray_norm, cv2.COLOR_GRAY2RGB)

                    # Сохранение
                    output_file = os.path.join(output_dir, os.path.splitext(img_name)[0] + ".jpg")
                    cv2.imwrite(output_file, rgb_image)  # автоматически как RGB

                    print(f"[INFO IMG] Сохранено {output_file}")

                except Exception as e:
                    print(f"[ERROR IMG] Ошибка при обработке {img_path}: {str(e)}")

    def _load_dataset(self, dataset_converted_path):
        encodings = []
        names = []

        for student_name in os.listdir(dataset_converted_path):
            student_dir = os.path.join(dataset_converted_path, student_name)
            if not os.path.isdir(student_dir):
                continue

            for img_name in os.listdir(student_dir):
                img_path = os.path.join(student_dir, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"[WARNING IMG] Не удалось загрузить {img_path}")
                    continue

                print(f"[DEBUG] {img_path}: shape={image.shape}, dtype={image.dtype}")
                if len(image.shape) != 3 or image.shape[2] not in [3, 4]:
                    print(f"[ERROR IMG] Неподдерживаемый формат изображения {img_path}: shape={image.shape}")
                    continue
                if image.shape[2] == 4:
                    print(f"[INFO IMG] Убираем альфа-канал для {img_path}")
                    image = image[:, :, :3]

                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                rgb_image = np.array(rgb_image, dtype=np.uint8)
                print(f"[DEBUG IMG] После обработки: shape={rgb_image.shape}, dtype={rgb_image.dtype}")

                try:
                    boxes = face_recognition.face_locations(rgb_image)
                    if len(boxes) == 0:
                        print(f"[INFO] Лицо не найдено на {img_path}")
                        continue
                    encoding = face_recognition.face_encodings(rgb_image, boxes)[0]
                    encodings.append(encoding)
                    names.append(student_name)
                except RuntimeError as e:
                    print(f"[ERROR IMG] Ошибка обработки {img_path}: {str(e)}")
                    continue

        return encodings, names

    def retrain_model(self):
        # конвертация
        self._convert_image()

        # загрузка данных
        known_encs, known_names = self._load_dataset(DATASET_CONVERTED_DIR)
        if not known_encs:
            raise ValueError("Нет изображений для обучения. Добавьте фото в папку датасета.")

        # обучение
        clf = SVC(probability=True, kernel='linear')
        try:
            clf.fit(known_encs, known_names)
        except Exception as e:
            raise RuntimeError(f"Ошибка при обучении модели: {e}")

        # сохр старую
        try:
            archive_file(FACE_MODEL_PATH, FACE_MODEL_PATH, create_file=False)
        except Exception as e:
            print(f"[WARNING IMG] Не удалось архивировать старую модель: {e}")

        # сохр новую
        try:
            with open(FACE_MODEL_PATH, "wb") as f:
                pickle.dump(clf, f)
        except Exception as e:
            raise IOError(f"Не удалось сохранить новую модель: {e}")

        # обновляем работающий экземпляр
        self.face_model = clf
        return True
