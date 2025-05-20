from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QDialog, QFormLayout, QDialogButtonBox, QDateEdit, QLineEdit,
    QVBoxLayout, QInputDialog, QMessageBox, QTimeEdit, QCheckBox, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer, QDate, QTime
from PyQt5.QtGui import QImage, QPixmap, QPainter, QBrush, QFont, QPen
import os
import cv2
import json
import sounddevice as sd
from threading import Thread
from datetime import datetime

from image import ImageProcessor
from audio import AudioProcessor
from load import dict_get_or_set
from conf import SETTING_JSON_PATH
from stats import get_stats


class SettingsWindow(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        draw_conf = self.parent.image_processor.draw_conf

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

        # Кнопка Информация
        info_btn = QPushButton("Информация", self)
        info_btn.clicked.connect(
            lambda: QMessageBox.information(
                self,
                "Информация",
                f"Настоящая программа сделана в рамкой дипломного проекта по направлению Информатика (Программирование)\n"
                f"Программа получает доступ к выбранным интерфейчас: камере и микрофону\n"
                f"После этого возможен сбор статистики:\n"
                f"[FACE] Распознование лица\n"
                f"[YOLO] Распознование объектов\n"
                f"[VOSK] Распознвоание слов\n"
                f"[YAMN] Распознование звуков\n"
                f"Собранная статистика участвует в оценке эффективности работника за выбранный период\n"
                f"Программа выводит % собранной статистики за указанное временное окно "
                f"и Коэффециэнт работоспособности сотрудника, исходя из собранной статистики."
            )
        )
        layout.addWidget(info_btn)

        # Группа для чекбоксов
        grp = QGroupBox("Настройки отрисовки", self)
        grp_layout = QVBoxLayout(grp)
        # Чекбоксы: для каждого ключа — своё описание
        for key, text in (
            ("face",  "Отрисовывать лица FACE"),
            ("yolo",  "Отрисовывать объекты YOLO"),
            ("vosk",  "Отрисовывать слова VOSK"),
            ("yamn",  "Отрисовывать звуки YAMNET")
        ):
            # загружаем текущее значение (или True по умолчанию)
            checked = dict_get_or_set(draw_conf, key, True)
            cb = QCheckBox(text, self)
            cb.setChecked(checked)
            # при переключении сохраняем в draw_conf и в settings
            cb.stateChanged.connect(lambda state, k=key: self._on_draw_toggle(k, state))
            grp_layout.addWidget(cb)
        layout.addWidget(grp)

        # Кнопка переобучени
        retrain_btn = QPushButton("Начать переобучение\nраспознования лиц", self)
        retrain_btn.clicked.connect(self.on_retrain_clicked)
        layout.addWidget(retrain_btn)

        # Кнопка статистика
        stats_btn = QPushButton("Узнать статистику рабочего\n", self)
        stats_btn.clicked.connect(self.on_stats_clicked)
        layout.addWidget(stats_btn)

        self.setLayout(layout)

    def _on_draw_toggle(self, key: str, state: int):
        self.parent.image_processor.draw_conf[key] = bool(state)
        self.parent.save_settings()

    def on_retrain_clicked(self):
        if self.parent.timer.isActive():
            QMessageBox.critical(self, "Error",
                f"Сначала остановите выполнение сбора статистики"
            )
            return

        # Спрашиваем подтверждение
        reply = QMessageBox.question(self, "Подтвердить переобучение", "Начать переобучение?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        try:
            self.parent.image_processor.retrain_model()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка переобучения", str(e))
        else:
            QMessageBox.information(self, "Успех", "Модель успешно переобучена!")

    def on_stats_clicked(self):
        if self.parent.timer.isActive():
            QMessageBox.critical(self, "Error",
                f"Сначала остановите выполнение сбора статистики"
            )
            return

        stats_conf = dict_get_or_set(self.parent.settings, "stats_conf", {})

        dlg = QDialog(self)
        dlg.setWindowTitle("Введите период статистики")
        form = QFormLayout(dlg)

        # Получаем старые значения из stats_conf
        name_old = dict_get_or_set(stats_conf, "name_old", "Name")
        date_old = dict_get_or_set(stats_conf, "date_old", datetime.today().date().isoformat())
        start_old = dict_get_or_set(stats_conf, "start_old", datetime.now().time().replace(microsecond=0).isoformat())
        end_old = dict_get_or_set(stats_conf, "end_old", datetime.now().time().replace(microsecond=0).isoformat())

        # Поля формы
        name_edit = QLineEdit(dlg)
        name_edit.setText(name_old)
        form.addRow("Имя сотрудника:", name_edit)

        date_edit = QDateEdit(dlg)
        date_edit.setCalendarPopup(True)
        date_edit.setDate(QDate.fromString(date_old, "yyyy-MM-dd"))
        form.addRow("Дата:", date_edit)

        start_edit = QTimeEdit(dlg)
        start_edit.setDisplayFormat("HH:mm:ss")
        start_edit.setTime(QTime.fromString(start_old, "HH:mm:ss"))
        form.addRow("Время начала:", start_edit)

        end_edit = QTimeEdit(dlg)
        end_edit.setDisplayFormat("HH:mm:ss")
        end_edit.setTime(QTime.fromString(end_old, "HH:mm:ss"))
        form.addRow("Время окончания:", end_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        form.addRow(buttons)

        def reset_fields():
            """Восстанавливает поля из stats_conf."""
            name_edit.setText(stats_conf["name_old"])
            date_edit.setDate(QDate.fromString(stats_conf["date_old"], "yyyy-MM-dd"))
            start_edit.setTime(QTime.fromString(stats_conf["start_old"], "HH:mm:ss"))
            end_edit.setTime(QTime.fromString(stats_conf["end_old"], "HH:mm:ss"))

        def accept():
            name = name_edit.text().strip()
            if not name:
                QMessageBox.warning(dlg, "Ошибка", "Введите имя сотрудника")
                reset_fields()
                return

            d = date_edit.date().toPyDate()
            t0 = start_edit.time().toPyTime()
            t1 = end_edit.time().toPyTime()
            if datetime.combine(d, t0) >= datetime.combine(d, t1):
                QMessageBox.warning(dlg, "Ошибка", "Время начала должно быть раньше времени окончания")
                reset_fields()
                return

            # Сохраняем новые значения
            self.stat_name = name
            self.stat_period = (d, t0, t1)

            stats_conf["name_old"] = name
            stats_conf["date_old"] = d.isoformat()
            stats_conf["start_old"] = t0.strftime("%H:%M:%S")
            stats_conf["end_old"] = t1.strftime("%H:%M:%S")
            self.parent.save_settings()

            dlg.accept()

        buttons.accepted.connect(accept)
        buttons.rejected.connect(dlg.reject)

        result = dlg.exec_()

        # Если пользователь нажал OK и всё прошло в accept — запускаем расчёт
        if result == QDialog.Accepted:
            percent_stats, percent_work = get_stats(self.parent.settings, self.stat_name, self.stat_period)
            print(f"[STATS] Собрано {percent_stats}%, Работоспособность {percent_work}")
            # Формируем строки для даты и времени
            d, t0, t1 = self.stat_period
            date_str = d.isoformat()
            start_str = t0.strftime("%H:%M:%S")
            end_str = t1.strftime("%H:%M:%S")

            # Собираем сообщение
            msg = (
                f"По сотруднику {self.stat_name} за {date_str} "
                f"в период {start_str}–{end_str} собрано {percent_stats:.1f}% статистики.\n"
                f"Коэффициент работоспособности сотрудника составил {percent_work:.2f}%."
            )

            # Показываем «краси&вую плашку»
            QMessageBox.information(
                self,
                "Результаты статистики",
                msg
            )

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.window_of_settings = None

        # Загрузка JSON настроек
        self.settings_path = SETTING_JSON_PATH
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, "r", encoding="utf-8") as f:
                    self.settings = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                # файл либо битый, либо вдруг удалился между exists() и open()
                self.settings = dict()
        else:
            self.settings = dict()

        # Заголовок и геометрия окна
        self.win_conf = dict_get_or_set(self.settings, "Window", {})
        self.setWindowTitle(dict_get_or_set(self.win_conf, "name", "WathGarg AI"))
        screen = QApplication.primaryScreen().availableGeometry()
        h = dict_get_or_set(self.win_conf, "h", 0.6)
        w = dict_get_or_set(self.win_conf, "w", 0.4)
        self.resize(int(screen.width() * w), int(screen.height() * h))
        self.move((screen.width() - self.width()) // 2,
                  (screen.height() - self.height()) // 2)

        # Обработчики
        self.image_processor = ImageProcessor(self.settings)
        self.audio_processor = AudioProcessor(self.settings)

        # Вычисление геометрии
        self.margin = int(min(self.width(), self.height()) * 0.03)
        self.btn_h = int(min(self.width(), self.height()) * 0.15)
        self.win_w, self.win_h = self.width(), self.height()
        self.bottom_y = self.win_h - self.btn_h - self.margin

        # Поворот
        self.rotation = 0  # угол в градусах: 0,90,180,270
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
        print(f"[UI] Started")

    def _apply_rotation(self, frame):
        if self.rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _rotate_left(self):
        self.rotation = (self.rotation - 90) % 360
        print(f"[UI] Rotation set to {self.rotation}")

    def _rotate_right(self):
        self.rotation = (self.rotation + 90) % 360
        print(f"[UI] Rotation set to {self.rotation}")

    def save_settings(self):
        try:
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=4)
            print(f"[SETTINGS] Сохранены в {self.settings_path}")
        except Exception as e:
            print(f"[SETTINGS ERROR] Не удалось сохранить настройки: {e}")

    def _create_buttons(self):
        m, b, W, y = self.margin, self.btn_h, self.win_w, self.bottom_y

        # Стандартные кнопки
        self.btn_left = QPushButton("⟲", self)
        self.btn_right = QPushButton("⟳", self)
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
        cam_idx = dict_get_or_set(self.settings, "camera", 0)
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
        mic_idx = dict_get_or_set(self.settings, "microphone", 0)
        if mic_idx < len(mic_names):
            mic_display = mic_names[mic_idx]
        else:
            mic_display = mic_names[0] if mic_names else "0: Микрофон 0"
        self.btn_microphone = QPushButton(f"Выбрать микрофон\n{mic_display}", self)

        # Устанавливаем высоту кнопок
        for btn in (self.btn_left, self.btn_right,
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
        self.btn_left.setGeometry(xs[0], y, b, b)
        self.btn_right.setGeometry(xs[1], y, b, b)
        self.btn_camera .setGeometry(xs[2], y, c, b)
        self.btn_microphone.setGeometry(xs[3], y, c, b)
        self.btn_start.setGeometry(xs[4], y, b, b)
        self.btn_settings.setGeometry(xs[5], y, b, b)

        # Поднимаем над видео
        for btn in (self.btn_left, self.btn_right,
                    self.btn_camera, self.btn_microphone,
                    self.btn_start, self.btn_settings):
            btn.raise_()

        # Привязка событий
        self.btn_left.clicked.connect(self._rotate_left)
        self.btn_right.clicked.connect(self._rotate_right)
        self.btn_camera.clicked.connect(self._select_camera)
        self.btn_microphone.clicked.connect(self._select_microphone)
        self.btn_start.clicked.connect(self._toggle_processing)
        self.btn_settings.clicked.connect(self._open_settings)

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

        current = self.settings.get("camera", idxs[0])
        current_name = next((n for n in names if int(n.split(':')[0]) == current), names[0])
        choice, ok = QInputDialog.getItem(
            self, "Выбор камеры", "Устройства:", names,
            names.index(current_name), False
        )
        if not ok:
            return
        # Новый и старый индексы
        new_index = int(choice.split(':')[0])
        old_index = self.image_processor.camera_index
        # Останавливаем текущий поток (если был)
        self.image_processor.stop_camera()
        # Применяем новые настройки
        self.settings["camera"] = new_index
        self.image_processor.settings["camera"] = new_index
        self.image_processor.camera_index = new_index

        new_name = choice
        old_name = next((n for n in names if int(n.split(':')[0]) == old_index), f"{old_index}")
        # Если видео уже работало — пробуем сразу запустить новую камеру
        if self.timer.isActive():
            if not self.image_processor.start_camera():
                # Откатываем все настройки
                self.settings["camera"] = old_index
                self.image_processor.settings["camera"] = old_index
                self.image_processor.camera_index = old_index
                # Пытаемся вернуть старую камеру
                if not self.image_processor.start_camera():
                    QMessageBox.critical(
                        self, "Критическая ошибка",
                        f"Не удалось ни открыть новую камеру «{new_name}»"
                        f"ни вернуть старую «{old_name}»."
                    )
                    self._stor_processing()
                else:
                    QMessageBox.critical(
                        self, "Ошибка смены камеры",
                        f"Не удалось открыть камеру «{new_name}»"
                        f"вернулась старая «{old_name}»."
                    )
                return

        self.btn_camera.setText(f"Выбрать камеру\n{choice}")
        print(f"[UI] Выбрана камера: {choice}")
        self.save_settings()

    def _select_microphone(self):
        import sounddevice as sd
        devs = sd.query_devices()
        inputs = [d for d in devs if d['max_input_channels'] > 0]
        names = [f"{i}: {d['name']}" for i, d in enumerate(inputs)]
        if not names:
            QMessageBox.warning(self, "Нет микрофонов", "Не найдено ни одного микрофона")
            return

        current = self.settings.get("microphone", 0)
        current_name = next((n for n in names if int(n.split(':')[0]) == current), names[0])
        choice, ok = QInputDialog.getItem(
            self, "Выбор микрофона", "Устройства:", names,
            names.index(current_name), False
        )
        if not ok:
            return

        # Новый и старый индексы
        new_idx = int(choice.split(':')[0])
        old_idx = self.audio_processor.mic_index
        # Останавливаем старый поток (если он был запущен)
        self.audio_processor.stop_microphone()
        # Применяем новые настройки
        self.settings["microphone"] = new_idx
        self.audio_processor.settings["microphone"] = new_idx
        self.audio_processor.mic_index = new_idx
        # Получаем человекочитаемые имена
        new_name = choice
        old_name = next((n for n in names if int(n.split(':')[0]) == old_idx), f"{old_idx}")

        # Если аудио уже работало — пробуем сразу запустить новый микрофон
        if getattr(self.audio_processor, "stream", None):
            if not self.audio_processor.start_microphone():
                # Откатываем настройки на старые
                self.settings["microphone"] = old_idx
                self.audio_processor.settings["microphone"] = old_idx
                self.audio_processor.mic_index = old_idx
                # Пытаемся вернуть старый микрофон
                if not self.audio_processor.start_microphone():
                    QMessageBox.critical(
                        self,
                        "Критическая ошибка",
                        f"Не удалось ни открыть новый микрофон «{new_name}»\n"
                        f"ни вернуть старый «{old_name}»."
                    )
                else:
                    QMessageBox.critical(
                        self,
                        "Ошибка смены микрофона",
                        f"Не удалось открыть микрофон «{new_name}»\n"
                        f"вернулся старый «{old_name}»."
                    )
                return

        self.btn_microphone.setText(f"Выбрать микрофон\n{new_name}")
        print(f"[UI] Выбран микрофон: {new_name}")
        self.save_settings()

    def _stor_processing(self):
        self.timer.stop()
        self.image_processor.stop_camera()
        self.audio_processor.stop_processing()
        self.video_label.clear()
        print("[UI PROC] Остановлены обработка видео и аудио")

    def _toggle_processing(self):
        # Переключение таймера
        if not self.timer.isActive():
            # Стартуем видео
            try:
                self.image_processor.start_camera()
            except Exception as e:
                QMessageBox.critical(self, "Ошибка камеры", str(e))
                return
            # Стартуем аудио
            try:
                self.audio_processor.start_microphone()
            except Exception as e:
                QMessageBox.critical(self, "Ошибка микрофона", str(e))
                return
            self.audio_processor.running_recognition = True
            if not self.audio_processor.running_recognition_thread:
                Thread(target=self.audio_processor.proc_audio, daemon=True).start()

            self.timer.start(30)
            print("[UI PROC] Запущены обработка видео и аудио")
        else:
            self._stor_processing()

    def _open_settings(self):
        if not self.window_of_settings:
            self.window_of_settings = SettingsWindow(self)
        self.window_of_settings.show()

    def _update_frame(self):
        # Видеопоток
        ret, frame = self.image_processor.get_frame()
        if not ret:
            return
        frame = self._apply_rotation(frame)

        # Обработка изображения (возвращает кадр и словарь с результатами)
        processed_frame, detect_results = self.image_processor.proc_image(frame)

        # Конвертация для Qt
        rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt).scaled(
            self.video_label.size(), Qt.KeepAspectRatio
        )

        # painter
        painter = QPainter(pix)
        # Индикатор звука в левом верхнем углу
        painter.setPen(Qt.NoPen)
        # выбираем цвет по флагу audio_active
        color = Qt.green if self.audio_processor.audio_active else Qt.lightGray
        painter.setBrush(QBrush(color))
        r = 10
        margin = 5
        w_pix = pix.width()
        x = w_pix - margin - 2*r
        painter.drawEllipse(x, margin, 2*r, 2*r)

        # последнюяя фраза VOSK
        if self.image_processor.draw_conf["vosk"]:
            text = self.audio_processor.vosk_text_buffer or ""
            if text:
                # шрифт
                font = QFont()
                font.setPointSize(14)
                painter.setFont(font)

                # вычисляем координаты
                margin = 8
                w_pix = pix.width()
                h_pix = pix.height()
                # рисуем текст над самым низом
                metrics = painter.fontMetrics()
                text_width = metrics.horizontalAdvance(text)
                text_height = metrics.height()
                x = margin
                y = h_pix - margin

                # чёрная обводка - текст четырьмя смещениями
                pen = QPen(Qt.black)
                pen.setWidth(2)
                painter.setPen(pen)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    painter.drawText(x + dx, y + dy, text)

                # белая заливка
                painter.setPen(Qt.white)
                painter.drawText(x, y, text)

        # события YAMNet
        if self.image_processor.draw_conf["yamn"]:
            lines = self.audio_processor.yamnet_text_buffer  # список строк
            if lines:
                # шрифт
                font = QFont()
                font.setPointSize(18)
                painter.setFont(font)
                metrics = painter.fontMetrics()

                margin = 8
                h_line = metrics.height()

                for i, line in enumerate(lines):
                    x = margin
                    y = margin + i * h_line + metrics.ascent()

                    # чёрная обводка - текст в четырёх смещениях
                    pen = QPen(Qt.white)
                    pen.setWidth(2)
                    painter.setPen(pen)
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        painter.drawText(x + dx, y + dy, line)

                    # белая заливка
                    painter.setPen(Qt.black)
                    painter.drawText(x, y, line)
        painter.end()

        self.video_label.setPixmap(pix)
