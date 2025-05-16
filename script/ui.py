from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QDialog,
    QVBoxLayout, QInputDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import json
import sounddevice as sd

from face import FaceProcessor
from audio import AudioProcessor
from load import check_exist, dict_get_or_set
from conf import setting_json_path


class SettingsWindow(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        # Заголовок
        self.setWindowTitle("Настройки")
        # Половина размера родительского окна
        w = int(parent.width() * 0.7)
        h = int(parent.height() * 0.7)
        self.resize(w, h)
        # Центрируем над родительским
        x = parent.x() + (parent.width() - w) // 2
        y = parent.y() + (parent.height() - h) // 2
        self.move(x, y)

        # Компоновка
        layout = QVBoxLayout(self)

        # Кнопка «Информация»
        info_btn = QPushButton("Информация", self)
        info_btn.clicked.connect(
            lambda: QMessageBox.information(
                self,
                "Информация",
                "TO DO..."
            )
        )
        layout.addWidget(info_btn)

        retrain_btn = QPushButton("Начать переобучение\nраспознования лиц", self)
        retrain_btn.clicked.connect(self.on_retrain_clicked)
        layout.addWidget(retrain_btn)

        self.setLayout(layout)

    def on_retrain_clicked(self):
        try:
            self.parent.face_processor.retrain_model()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка переобучения", str(e))
        else:
            QMessageBox.information(self, "Успех", "Модель успешно переобучена!")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Загрузка JSON настроек
        check_exist(setting_json_path)
        with open(setting_json_path, "r", encoding="utf-8") as f:
            self.setting_json = json.load(f)
        self.settings_path = setting_json_path

        # Заголовок и геометрия окна
        win_conf = dict_get_or_set(self.setting_json, "Window", {})
        self.setWindowTitle(dict_get_or_set(win_conf, "name", "Supervisor"))
        screen = QApplication.primaryScreen().availableGeometry()
        h = dict_get_or_set(win_conf, "h", 0.5)
        w = dict_get_or_set(win_conf, "w", 0.4)
        self.resize(int(screen.width() * w), int(screen.height() * h))
        self.move((screen.width() - self.width()) // 2,
                  (screen.height() - self.height()) // 2)

        # Инициализация процессоров
        self.face_processor = FaceProcessor(self.setting_json)
        self.audio_processor = AudioProcessor(self.setting_json)

        # Вычисление геометрии
        self.margin = int(min(self.width(), self.height()) * 0.03)
        self.btn_h = int(min(self.width(), self.height()) * 0.15)
        self.win_w, self.win_h = self.width(), self.height()
        self.bottom_y = self.win_h - self.btn_h - self.margin

        # QLabel для видео
        self.video_label = QLabel(self)
        self.video_label.setGeometry(
            0, 0,
            self.win_w,
            self.win_h - self.btn_h - self.margin
        )
        self.video_label.setAlignment(Qt.AlignCenter)

        # Создание кнопок
        self._create_buttons()

        # Таймер для обновления видео
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

        # список камер
        try:
            from pygrabber.dshow_graph import FilterGraph
            cams = FilterGraph().get_input_devices()
            print("[VIDEO] Доступные камеры (DirectShow):")
            for i, name in enumerate(cams):
                print(f"  index={i}, name={name}")
        except ImportError:
            print("[VIDEO] pygrabber не установлен, ищем по индексам:")
            for i in range(10):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        print(f"  index={i}, name=Камера {i}")
                    cap.release()
        # список микрофонов
        devs = sd.query_devices()
        inputs = [d for d in devs if d['max_input_channels'] > 0]
        print("[AUDIO] Доступные устройства ввода:")
        for i, d in enumerate(inputs):
            print(f"  [{i}] index={d['index']} name={d['name']}")

    def _save_settings(self):
        try:
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(self.setting_json, f, ensure_ascii=False, indent=4)
            print(f"[SETTINGS] Сохранены в {self.settings_path}")
        except Exception as e:
            print(f"[SETTINGS ERROR] Не удалось сохранить настройки: {e}")

    def _create_buttons(self):
        m, b, W, y = self.margin, self.btn_h, self.win_w, self.bottom_y

        # Стандартные кнопки
        self.btn_new1 = QPushButton("Новая1", self)
        self.btn_new2 = QPushButton("Новая2", self)
        self.btn_start = QPushButton("Старт/Стоп", self)
        self.btn_settings = QPushButton("Настройки", self)

        # Кнопка выбора камеры
        # Получаем список устройств и отображаем текущий по индексу
        try:
            from pygrabber.dshow_graph import FilterGraph
            cams = FilterGraph().get_input_devices()
        except ImportError:
            cams = []
            for i in range(10):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        cams.append(f"Камера {i}")
                cap.release()
        # Формируем имена 'i: Name'
        cam_names = [f"{i}: {name}" for i, name in enumerate(cams)]
        cam_idx = dict_get_or_set(self.setting_json, "camera", 0)
        if cam_idx < len(cam_names):
            cam_display = cam_names[cam_idx]
        else:
            cam_display = cam_names[0] if cam_names else "0: Камера 0"
        self.btn_camera = QPushButton(f"Выбрать камеру\n{cam_display}", self)

        # Кнопка выбора микрофона
        import sounddevice as sd
        devs = sd.query_devices()
        mics = [d['name'] for d in devs if d['max_input_channels'] > 0]
        mic_names = [f"{i}: {name}" for i, name in enumerate(mics)]
        mic_idx = dict_get_or_set(self.setting_json, "microphone", 0)
        if mic_idx < len(mic_names):
            mic_display = mic_names[mic_idx]
        else:
            mic_display = mic_names[0] if mic_names else "0: Микрофон 0"
        self.btn_microphone = QPushButton(f"Выбрать микрофон\n{mic_display}", self)

        # Устанавливаем высоту кнопок
        for btn in (self.btn_new1, self.btn_new2,
                    self.btn_camera, self.btn_microphone,
                    self.btn_start, self.btn_settings):
            btn.setFixedHeight(b)

        # Ширина центральных двух кнопок
        central_total = W - 4*b - 7*m
        c = max(int(central_total / 2), 0)

        xs = [
            m,
            m + b + m,
            m + 2*(b + m),
            m + 2*(b + m) + c + m,
            m + 2*(b + m) + 2*(c + m),
            m + 2*(b + m) + 2*(c + m) + b + m
        ]

        # Помещаем кнопки
        self.btn_new1    .setGeometry(xs[0], y, b, b)
        self.btn_new2    .setGeometry(xs[1], y, b, b)
        self.btn_camera  .setGeometry(xs[2], y, c, b)
        self.btn_microphone.setGeometry(xs[3], y, c, b)
        self.btn_start   .setGeometry(xs[4], y, b, b)
        self.btn_settings.setGeometry(xs[5], y, b, b)

        # Поднимаем над видео
        for btn in (self.btn_new1, self.btn_new2,
                    self.btn_camera, self.btn_microphone,
                    self.btn_start, self.btn_settings):
            btn.raise_()

        # Привязка событий
        self.btn_new1     .clicked.connect(lambda: print("Новая1 нажата"))
        self.btn_new2     .clicked.connect(lambda: print("Новая2 нажата"))
        self.btn_camera   .clicked.connect(self._select_camera)
        self.btn_microphone.clicked.connect(self._select_microphone)
        self.btn_start    .clicked.connect(self._toggle_processing)
        self.btn_settings .clicked.connect(self._open_settings)

    def resizeEvent(self, event):
        self.margin = int(min(self.width(), self.height()) * 0.03)
        self.btn_h = int(min(self.width(), self.height()) * 0.15)
        self.win_w, self.win_h = self.width(), self.height()
        self.bottom_y = self.win_h - self.btn_h - self.margin

        self.video_label.setGeometry(
            0, 0,
            self.win_w,
            self.win_h - self.btn_h - self.margin
        )
        self._create_buttons()
        super().resizeEvent(event)

    def _select_camera(self):
        try:
            from pygrabber.dshow_graph import FilterGraph
            cams = FilterGraph().get_input_devices()
            names = [f"{i}: {name}" for i, name in enumerate(cams)]
            idxs = list(range(len(cams)))
        except ImportError:
            names, idxs = [], []
            for i in range(10):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        names.append(f"{i}: Камера {i}")
                        idxs.append(i)
                cap.release()

        if not idxs:
            QMessageBox.warning(self, "Нет камер", "Не найдено ни одной камеры")
            return

        current = self.setting_json.get("camera", idxs[0])
        current_name = next((n for n in names if int(n.split(':')[0]) == current), names[0])
        choice, ok = QInputDialog.getItem(
            self, "Выбор камеры", "Устройства:", names,
            names.index(current_name), False
        )
        if not ok:
            return

        sel_index = int(choice.split(':')[0])
        self.setting_json["camera"] = sel_index
        self.face_processor.settings["camera"] = sel_index
        self.face_processor.camera_index = sel_index
        self.btn_camera.setText(f"Выбрать камеру\n{choice}")
        print(f"[UI] Выбрана камера: {choice}")
        self._save_settings()

    def _select_microphone(self):
        import sounddevice as sd
        devs = sd.query_devices()
        inputs = [d for d in devs if d['max_input_channels'] > 0]
        names = [f"{i}: {d['name']}" for i, d in enumerate(inputs)]
        if not names:
            QMessageBox.warning(self, "Нет микрофонов", "Не найдено ни одного микрофона")
            return

        current = self.setting_json.get("microphone", 0)
        current_name = next((n for n in names if int(n.split(':')[0]) == current), names[0])
        choice, ok = QInputDialog.getItem(
            self, "Выбор микрофона", "Устройства:", names,
            names.index(current_name), False
        )
        if not ok:
            return

        sel = int(choice.split(':')[0])
        self.setting_json["microphone"] = sel
        self.audio_processor.settings["microphone"] = sel
        self.audio_processor.mic_index = sel
        self.btn_microphone.setText(f"Выбрать микрофон\n{choice}")
        print(f"[UI] Выбран микрофон: {choice}")
        self._save_settings()

    def _toggle_processing(self):
        # Стартуем видео
        if not self.face_processor.start_camera():
            QMessageBox.critical(self, "Ошибка", "Не удалось загрузить модель или открыть камеру.")
            return
        # Стартуем аудио
        if not self.audio_processor.start_microphone():
            QMessageBox.critical(self, "Ошибка", "Не удалось открыть микрофон.")
            return
        # Переключение таймера
        if not self.timer.isActive():
            self.timer.start(30)
            print("[UI] Запущены видео и аудио")
        else:
            self.timer.stop()
            self.face_processor.stop_camera()
            self.audio_processor.stop_microphone()
            self.video_label.clear()
            print("[UI] Остановлены видео и аудио")

    def _open_settings(self):
        self.settings_win = SettingsWindow(self)
        self.settings_win.show()

    def _update_frame(self):
        # Видеопоток
        ret, frame = self.face_processor.get_frame()
        if not ret:
            return
        img, _ = self.face_processor.process_frame(frame)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt).scaled(
            self.video_label.size(), Qt.KeepAspectRatio
        )
        self.video_label.setPixmap(pix)

        # Audio-обработчик (запуск recognize_loop в фоновом потоке один раз)
        if not getattr(self, "_audio_thread_started", False):
            from threading import Thread
            Thread(target=self.audio_processor.recognize_loop, daemon=True).start()
            self._audio_thread_started = True

