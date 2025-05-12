from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout,
    QWidget, QLabel, QDialog, QMessageBox
)
from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2

from face import FaceProcessor
from funk import dict_get_or_set


class PopupWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Это всплывающее окно")

        # размер окна (50% от размеров родителя)
        parent_size = parent.size()
        self.window_width = int(parent_size.width() * 0.5)
        self.window_height = int(parent_size.height() * 0.5)
        self.setFixedSize(QSize(self.window_width, self.window_height))

        # содержимое окна
        message = QLabel("Закройте это окно кнопкой или крестиком!")
        close_button = QPushButton("Закрыть")
        close_button.clicked.connect(self.accept)  # Закрытие окна через accept()
        layout = QVBoxLayout()
        layout.addWidget(message)
        layout.addWidget(close_button)
        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self, setting_json, face_model):
        super().__init__()

        # Заголовок и геометрия окна
        window_conf = dict_get_or_set(setting_json, "Window", {})
        self.setWindowTitle(dict_get_or_set(window_conf, "name", "Supervisor"))
        screen = QApplication.primaryScreen().availableGeometry()
        h = dict_get_or_set(window_conf, "h", 0.5)
        w = dict_get_or_set(window_conf, "w", 0.4)
        self.resize(int(screen.width() * w), int(screen.height() * h))
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 2
        )

        # Полный JSON передаём в FaceProcessor
        self.face_processor = FaceProcessor(face_model, setting_json)

        # Параметры для расположения
        self.margin = int(min(self.width(), self.height()) * 0.03)
        self.btn_size = int(min(self.width(), self.height()) * 0.15)
        self.bottom_y = self.height() - self.btn_size - self.margin

        # Виджет для видео (под кнопками)
        self.video_label = QLabel(self)
        self.video_label.setGeometry(
            0, 0,
            self.width(),
            self.height() - self.btn_size - self.margin
        )
        self.video_label.setAlignment(Qt.AlignCenter)

        # Создаём 5 кнопок в один ряд внизу
        # 2 левые — новые, центральная — выбор камеры, 2 правые — настройки и старт/стоп
        self.btn_new1 = QPushButton("Новая1", self)
        self.btn_new2 = QPushButton("Новая2", self)
        self.btn_camera = QPushButton("Камера", self)
        self.btn_settings = QPushButton("Настройки", self)
        self.btn_start = QPushButton("Старт/Стоп", self)

        for btn in (self.btn_new1, self.btn_new2,
                    self.btn_camera,
                    self.btn_settings, self.btn_start):
            btn.setFixedHeight(self.btn_size)

        # Располагаем кнопки внизу
        self._reposition_buttons()

        # Привязываем пока что все новые кнопки к выводам в консоль
        self.btn_new1.clicked.connect(lambda: print("Новая1 нажата"))
        self.btn_new2.clicked.connect(lambda: print("Новая2 нажата"))
        self.btn_camera.clicked.connect(lambda: print("Кнопка Камера нажата"))
        # Настройки пока через всплывающее окно
        self.btn_settings.clicked.connect(self._open_settings)
        # Старт/Стоп запускает и останавливает поток
        self.btn_start.clicked.connect(self._toggle_camera)

        # Таймер обновления кадра
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

    def _reposition_buttons(self):
        m = self.margin
        b = self.btn_size
        W = self.width()
        # количество фиксированных кнопок: 4 * b + 6 * m
        central_w = W - (4 * b + 6 * m)
        # X-координаты:
        x0 = m
        x1 = m + b + m
        x2 = m + b + m + b + m  # = 3m + 2b
        x3 = x2 + central_w + m
        x4 = x3 + b + m
        y = self.bottom_y

        # Ширины:
        self.btn_new1.setFixedWidth(b)
        self.btn_new2.setFixedWidth(b)
        self.btn_settings.setFixedWidth(b)
        self.btn_start.setFixedWidth(b)
        self.btn_camera.setFixedWidth(central_w)

        # Устанавливаем геометрию:
        self.btn_new1.setGeometry(x0, y, b, b)
        self.btn_new2.setGeometry(x1, y, b, b)
        self.btn_camera.setGeometry(x2, y, central_w, b)
        self.btn_settings.setGeometry(x3, y, b, b)
        self.btn_start.setGeometry(x4, y, b, b)

        # Кнопки поверх видео
        for btn in (self.btn_new1, self.btn_new2,
                    self.btn_camera, self.btn_settings, self.btn_start):
            btn.raise_()

    def resizeEvent(self, event):
        # Пересчитаем параметры при изменении размера
        self.margin = int(min(self.width(), self.height()) * 0.03)
        self.btn_size = int(min(self.width(), self.height()) * 0.15)
        self.bottom_y = self.height() - self.btn_size - self.margin

        # Видео
        self.video_label.setGeometry(
            0, 0, self.width(),
            self.height() - self.btn_size - self.margin
        )
        # Кнопки
        for btn in (self.btn_new1, self.btn_new2,
                    self.btn_camera, self.btn_settings, self.btn_start):
            btn.setFixedHeight(self.btn_size)
        self._reposition_buttons()

        super().resizeEvent(event)

    def _open_settings(self):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout
        dlg = QDialog(self)
        dlg.setWindowTitle("Настройки")
        dlg.setFixedSize(self.width()//2, self.height()//2)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Здесь будут настройки"))
        dlg.exec_()

    def _toggle_camera(self):
        if not self.face_processor.video_capture:
            if self.face_processor.start_camera():
                self.timer.start(30)
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось открыть камеру")
        else:
            self.face_processor.stop_camera()
            self.timer.stop()
            self.video_label.clear()

    def _update_frame(self):
        ret, frame = self.face_processor.get_frame()
        if not ret:
            return
        img, _ = self.face_processor.process_frame(frame)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt).scaled(
            self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(pix)
