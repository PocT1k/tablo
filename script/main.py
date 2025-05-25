import tensorflow_hub as hub # НЕ УБИРАТЬ ЭТУ СТРОЧКУ!!! ЭТА ШТУКА ДОЛЖНА ПОДКЛЮЧАТЬСЯ В ФАЙЛЕ ТОЧКИ ВХОДА!!!
import sys
import random
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QThread, pyqtSignal


from ui import MainWindow
from conf import ICON_WINS_PATH, ICON_PANEL_PATH, ICON_TRY_PATH


class InitThread(QThread):
    hub
    error = pyqtSignal(str) # для фатальных исключений
    warning = pyqtSignal(list) # для накопленных предупреждений
    finished = pyqtSignal()

    def __init__(self, window):
        super().__init__(window)
        self.window = window

    def run(self):
        try:
            self.window.init_processors()
        except Exception as e:
            # Фатальная ошибка
            self.error.emit(str(e))
        else:
            # Все warnings из image и audio процессоров
            warns: list[str] = []
            for proc in (self.window.image_processor, self.window.audio_processor):
                if hasattr(proc, "init_warnings"):
                    warns.extend(proc.init_warnings)

            if warns:
                self.warning.emit(warns)
                # очищаем буферы предупреждений у процессоров
                for proc in (self.window.image_processor, self.window.audio_processor):
                    if hasattr(proc, "init_warnings"):
                        proc.init_warnings.clear()
        finally:
            self.finished.emit()

def main():
    # Установка AppUserModelID для иконки на панели задач (требуется для Windows 7+)
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('WatchGuard AI')

    app = QApplication(sys.argv)

    # Стиль
    app.setStyleSheet("""        
        /* Кнопок */
        QPushButton {
            background-color: #2e313a;
            color: #f8fafa;
            border: 2px solid #38738f;
            border-radius: 6px;
            padding: 4px 8px;
        }
        QPushButton:hover {
            background-color: #3dbbe4;
        }
        QPushButton:pressed {
            background-color: #3dbbe4;
            border-color: #2w313a;
        }
        
        /* Главного окна */
        QMainWindow {
            background-color: #d9d9d7;
        }
        /* Окна настроек */
        QSettingsWindow {
            background-color: #d9d9d7;
        }
    """)

    # Иконки всех окон
    wins_icon = QIcon(str(ICON_WINS_PATH))
    try_icon = QIcon(str(ICON_TRY_PATH))
    panel_icon = QIcon(str(ICON_PANEL_PATH))
    # Иконка на панели
    app.setWindowIcon(panel_icon)

    # Главное окно
    window = MainWindow()
    window.setWindowIcon(wins_icon) # Иконка главного окна
    window.show()


    init_thread = InitThread(window)
    init_thread.error.connect(lambda msg: QMessageBox.critical(
        window, "Ошибка при инициализации", msg
    ))

    init_thread.warning.connect(lambda warns: QMessageBox.warning(
        window,
        "Предупреждения при загрузке",
        "\n".join(warns)
    ))

    init_thread.finished.connect(lambda: (
        window._loader_timer.stop(),
        window.loader.hide()
    ))

    init_thread.start()

    # Анимация прогресса
    def tick():
        # пока thread жив — шагаем
        if init_thread.isRunning():
            v = window.loader.value() + random.randint(2, 5)
            window.loader.setValue(min(v, 100))
        else:
            # на всякий случай: вдруг сигналы шарахнут раньше — убедимся, что спрятали
            window._loader_timer.stop()
            window.loader.hide()

    window.loader.setValue(0)
    window.loader.show()
    window._loader_timer.timeout.connect(tick)
    window._loader_timer.start(100)  # например, 100ms

    sys.exit(app.exec_())  # цикл событий


if __name__ == "__main__":
    main()
