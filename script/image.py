import os
import pickle
import cv2
import numpy as np
import face_recognition
from sklearn.svm import SVC
from datetime import datetime
from ultralytics import YOLO
from PyQt5.QtWidgets import QMessageBox

from load import dict_get_or_set, archive_file, archive_file_by_date, write_attendance_dated
from conf import (FACE_MODEL_PATH, YOLO_MODEL_PATH, # модели
    FACE_ATTENDANCE_PATH, FACE_ATTENDANCE_OLD_DIR, YOLO_CLASSEC_PATH, YOLO_CLASSEC_OLD_DIR, # файлы логирования
    DATASET_CONVERTED_DIR, DATASET_RAW_DIR) # датасеты

class ImageProcessor:
    def __init__(self, settings):
        self.settings = settings
        # Интервал распознавания (секунды) и индекс камеры
        self.recognition_time = dict_get_or_set(self.settings, "image_time_recognition", 2)
        self.draw_conf = dict_get_or_set(self.settings, "draw_conf", {})
        self.draw_conf["face"] = dict_get_or_set(self.draw_conf, "face", True)
        self.draw_conf["yolo"] = dict_get_or_set(self.draw_conf, "yolo", True)
        self.draw_conf["vosk"] = dict_get_or_set(self.draw_conf, "vosk", True)
        self.draw_conf["yamn"] = dict_get_or_set(self.draw_conf, "yamn", True)
        self.camera_index = dict_get_or_set(self.settings, "camera", 0)
        print(f"[VIDEO] Окно распознования изображения = {self.recognition_time}с.")
        self.video_capture = None

        # Face
        self.load_face()
        archive_file_by_date(FACE_ATTENDANCE_PATH, FACE_ATTENDANCE_OLD_DIR, True)

        # YOLOv5
        self.yolo_threshold = dict_get_or_set(self.settings, "yolo_threshold", 0.7)
        self.yolo_classes = dict_get_or_set(self.settings, "yolo_classes", {})
        self.yolo_groups = {
            group: np.array(indices, dtype=int)
            for group, indices in self.yolo_classes.items()
            if indices  # пропускаем пустые списки
        }
        archive_file_by_date(YOLO_CLASSEC_PATH, YOLO_CLASSEC_OLD_DIR, True)
        self.yolo_ok = False
        try:
            if os.path.isfile(YOLO_MODEL_PATH):
                if not self.yolo_classes:
                    raise ValueError("yolo_classes не заданы в настройках")
                self.yolo_model = YOLO(YOLO_MODEL_PATH)
                self.yolo_ok = True
                print("[YOLO] Модель YOLO загружена успешно.")
            else:
                print("[YOLO ERROR] Модель распознования объектов не найдена")
                raise ValueError("Модель распознования объектов не найдена")
        except Exception as e:
            print("[YOLO ERROR] Не удалось загрузить YOLO-модель:", e)
            QMessageBox.warning(None, "Ошибка загрузки",
                                f'Не удалось загрузить YOLO: {e}')

        # Для кэширования результатов между распознаваниями
        self.last_detection_time = None
        self.faces_results = []
        self.items_results = []

    def load_face(self):
        self.face_threshold = dict_get_or_set(self.settings, "face_threshold", 0.6)
        self.face_scale = dict_get_or_set(self.settings, "face_scale", 0.5)
        self.inv_face_scale_index = 1.0 / self.face_scale  # обратный коэффициент
        print(f"[FACE] Порог распознования лиц = {self.face_threshold}")
        print(f"[FACE] Коэфециэнт масштабирования изображения для распознавания лиц = {self.face_scale} ({self.inv_face_scale_index})")

        self.face_ok = False
        try:
            with open(FACE_MODEL_PATH, "rb") as f:
                self.face_model = pickle.load(f)
            self.face_ok = True
            print("[FACE] Модель Face загружена успешно.")
        except Exception as e:
            print("[FACE ERROR] Не удалось загрузить Face:", e)
            QMessageBox.warning(None, "Ошибка загрузки",
                f'Не удалось загрузить Face: {e}')

        if not self.face_ok:
            print("[FACE] Модель распознования лиц не найдена")
            QMessageBox.warning(None, "Ошибка загрузки",
                                'Модуль распознования лиц FACE отключён\nМодель не найдена \nВы можетете новоую модель в "Настройки" - "Начать переобучение распознования лиц"')


    def start_camera(self):
        self.load_face()

        # попытка открыть выбранную камеру
        idx = self.camera_index
        print(f"[VIDEO] Попытка открыть камеру idx={idx}")
        self.video_capture = cv2.VideoCapture(idx)
        if not self.video_capture.isOpened():
            print(f"[VIDEO ERROR] Не удалось открыть камеру {idx}")
            raise ValueError("Не удалось открыть камеру")
        print(f"[VIDEO] Камера {idx} успешно открыта")

        return True

    def stop_camera(self):
        """Остановка видеопотока"""
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None

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

    def get_frame(self):
        if self.video_capture and self.video_capture.isOpened():
            return self.video_capture.read()
        return False, None

    def proc_face(self, frame):
        """
        return список словарей с ключами:
            'name' (строка),
            'threshold' (float),
            'location' (top, right, bottom, left).
        Не меняет исходный frame!
        """
        if not self.face_ok:
            return
        faces = []

        # Подготовка для детекции
        small = cv2.resize(frame, (0, 0), fx=self.face_scale, fy=self.face_scale)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locs = face_recognition.face_locations(rgb_small)
        encs = face_recognition.face_encodings(rgb_small, locs)

        # Собираем все имена в этом кадре
        names = []
        for (top, right, bottom, left), enc in zip(locs, encs):
            probs = self.face_model.predict_proba([enc])[0]
            idx = np.argmax(probs)
            name = self.face_model.classes_[idx]
            conf = probs[idx]
            if conf < self.face_threshold:
                name = "Unknown"
            names.append(name)

            # Приводим координаты к оригинальному размеру
            top = int(top * self.inv_face_scale_index)
            right = int(right * self.inv_face_scale_index)
            bottom = int(bottom * self.inv_face_scale_index)
            left = int(left * self.inv_face_scale_index)

            faces.append({
                "name": name,
                "threshold": float(conf),
                "location": (top, right, bottom, left)
            })

        write_attendance_dated(FACE_ATTENDANCE_PATH, names, None, "[FACE]")
        return faces

    def proc_yolo(self, frame):
        """
        return список словарей вида:
            'label' (строка),
            'threshold' (float),
            'location' (top, right, bottom, left).
        Не меняет исходный frame!
        """
        if not self.yolo_ok:
            return
        items = []
        results = self.yolo_model(frame, verbose=False)[0]

        # results.boxes.cls — индексы классов
        # results.boxes.conf — confidence
        # results.boxes.xyxy — [x1, y1, x2, y2]
        for cls, conf, box in zip(results.boxes.cls, results.boxes.conf, results.boxes.xyxy):
            idx = int(cls)
            if conf < self.yolo_threshold:
                continue
            # ищем все группы, в которых встречается этот idx
            for group, idx_arr in self.yolo_groups.items():
                if idx in idx_arr:
                    x1, y1, x2, y2 = map(int, box)
                    items.append({
                        "label": group,
                        "threshold": float(conf),
                        "location": (y1, x2, y2, x1)
                    })

        groups = sorted({item["label"] for item in items})
        write_attendance_dated(YOLO_CLASSEC_PATH, groups, None, "[YOLO]" )

        return items

    def proc_image(self, frame):
        # Проверяем, нужно ли делать детекцию в этом кадре
        now = datetime.now()
        do_detect = (
                self.last_detection_time is None or
                (now - self.last_detection_time).total_seconds() >= self.recognition_time
        )
        if do_detect:
            self.last_detection_time = now
            # детект лиц
            self.faces_results = self.proc_face(frame)
            # детект предметов
            self.items_results = self.proc_yolo(frame)

        frame_copy = frame.copy()
        # рисуем предметы (синяя)
        if self.draw_conf["yolo"]:
            for res in self.items_results:
                t, r, b, l = res["location"]
                cv2.rectangle(frame_copy, (l, t), (r, b), (255, 0, 0), 2)
                cv2.putText(
                    frame_copy,
                    f"{res['label']} ({res['threshold'] * 100:.1f}%)",
                    (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2
                )
        # рисуем лица (зелёная)
        if self.draw_conf["face"]:
            for res in self.faces_results:
                t, r, b, l = res["location"]
                cv2.rectangle(frame_copy, (l, t), (r, b), (0, 255, 0), 2)
                cv2.putText(
                    frame_copy,
                    f"{res['name']} ({res['threshold'] * 100:.1f}%)",
                    (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
                )

        return frame_copy, {"faces": self.faces_results, "items": self.items_results}
