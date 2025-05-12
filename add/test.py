import cv2
import numpy as np
from PyQt5.QtGui import QImage
from typing import Optional, Tuple

class VideoProcessor:
    def __init__(self):
        self.capture: Optional[cv2.VideoCapture] = None
        self.current_frame: Optional[np.ndarray] = None

    def start_capture(self, camera_index: int = 0) -> bool:
        """Запускает видеопоток с указанной камеры"""
        self.capture = cv2.VideoCapture(camera_index)
        return self.capture.isOpened()

    def stop_capture(self) -> None:
        """Останавливает видеопоток"""
        if self.capture:
            self.capture.release()
        self.capture = None

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Получает текущий кадр"""
        if self.capture and self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                self.current_frame = frame
            return ret, frame
        return False, None

    def process_frame(self, frame: np.ndarray) -> QImage:
        """Конвертирует кадр в QImage"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        return QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)

    def detect_faces(self, frame: np.ndarray) -> np.ndarray:
        """Обнаружение лиц (заглушка для реализации)"""
        # Здесь может быть ваша логика распознавания лиц
        return frame


from PyQt5.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QWidget,
                             QPushButton, QHBoxLayout, QMessageBox, QGridLayout, QStackedLayout)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap
#from video_processor import VideoProcessor


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # размеры экрана
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        self.window_width = int(screen_geometry.width() * 0.4)
        self.window_height = int(screen_geometry.height() * 0.4)
        self.resize(self.window_width, self.window_height)
        self.setWindowTitle("1")

        # Инициализация видео процессора
        self.video_processor = VideoProcessor()

        # Настройка UI
        self.setup_ui()

        # Таймер для обновления кадров
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def setup_ui(self):
        """Инициализация пользовательского интерфейса"""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Убираем стандартные отступы
        self.central_widget.setContentsMargins(0, 0, 0, 0)

        # Используем QGridLayout для наложения кнопок поверх видео
        self.main_layout = QGridLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # 1. Виджет для отображения видео (занимает всю площадь)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.main_layout.addWidget(self.video_label, 0, 0, 1, 1)  # Занимает всю сетку

        # 2. Создаем контейнер для кнопок поверх видео
        self.buttons_container = QWidget()
        self.buttons_container.setStyleSheet("background-color: transparent;")
        self.buttons_layout = QHBoxLayout(self.buttons_container)
        self.buttons_layout.setContentsMargins(0, 0, 10, 10)  # Отступы от краев
        self.buttons_layout.setSpacing(10)

        # Кнопка "Настройки"
        self.settings_button = QPushButton("Настройки")
        self.settings_button.setFixedSize(80, 80)
        self.settings_button.clicked.connect(self.show_settings)
        self.buttons_layout.addStretch()  # Растягиваем пространство слева
        self.buttons_layout.addWidget(self.settings_button)

        # Кнопка "Старт/Стоп"
        self.toggle_button = QPushButton("Старт")
        self.toggle_button.setFixedSize(80, 80)
        self.toggle_button.clicked.connect(self.toggle_camera)
        self.buttons_layout.addWidget(self.toggle_button)

        # Кнопка "Выбрать веб-камеру" (внизу)
        self.camera_select_button = QPushButton("Выбрать веб-камеру")
        self.camera_select_button.setFixedHeight(50)
        self.camera_select_button.clicked.connect(self.select_camera)

        # Добавляем контейнер с кнопками поверх видео
        self.main_layout.addWidget(self.buttons_container, 0, 0, 1, 1, Qt.AlignTop | Qt.AlignRight)

        # Добавляем кнопку выбора камеры отдельно (внизу)
        self.main_layout.addWidget(self.camera_select_button, 0, 0, 1, 1, Qt.AlignBottom)

    def resizeEvent(self, event):
        """Автоматически вызывается при изменении размера окна"""
        super().resizeEvent(event)
        # Обновляем изображение при изменении размера окна
        if hasattr(self, 'video_label') and self.video_label.pixmap():
            self.update_frame()

    def show_settings(self):
        """Показывает окно настроек"""
        settings_dialog = QMessageBox(self)
        settings_dialog.setWindowTitle("Настройки")
        settings_dialog.setText("Здесь будут настройки приложения")
        settings_dialog.exec_()

    def toggle_camera(self):
        """Переключает состояние камеры"""
        if not self.video_processor.capture:
            if self.video_processor.start_capture():
                self.toggle_button.setText("Стоп")
                self.timer.start(30)  # ~33 FPS
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось запустить камеру")
        else:
            self.video_processor.stop_capture()
            self.toggle_button.setText("Старт")
            self.timer.stop()
            self.video_label.clear()

    def select_camera(self):
        """Обработчик выбора камеры"""
        print("Выбрать веб-камеру (функционал будет реализован позже)")
        # Здесь будет логика выбора камеры

    def update_frame(self):
        """Обновляет отображаемый кадр"""
        ret, frame = self.video_processor.get_frame()
        if not ret:
            self.toggle_camera()
            return

        # Обработка кадра
        qt_image = self.video_processor.process_frame(frame)

        # Рассчитываем оптимальный размер для сохранения пропорций
        img_width = qt_image.width()
        img_height = qt_image.height()
        label_width = self.video_label.width()
        label_height = self.video_label.height()

        # Вычисляем коэффициенты масштабирования
        w_ratio = label_width / img_width
        h_ratio = label_height / img_height
        scale_factor = min(w_ratio, h_ratio)

        # Масштабируем изображение
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            int(img_width * scale_factor),
            int(img_height * scale_factor),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.video_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """Обработчик изменения размера окна"""
        super().resizeEvent(event)
        # Обновляем размер изображения при изменении окна
        if hasattr(self, 'video_label') and self.video_label.pixmap():
            self.video_label.setPixmap(self.video_label.pixmap().scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

    def closeEvent(self, event):
        """Обработчик закрытия окна"""
        self.video_processor.stop_capture()
        event.accept()


import sys
from PyQt5.QtWidgets import QApplication
#from main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())