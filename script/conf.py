from pathlib import Path

data_path = Path(Path(__file__).parent.parent / "data").expanduser().resolve()
setting_json_path = Path(data_path / "setting.json").expanduser().resolve()

# модели
model_path = Path(data_path / "model").expanduser().resolve()
# лица для обучения
face_path = Path(data_path / "face").expanduser().resolve()
dataset_to_convert_path = Path(face_path / "dataset").expanduser().resolve()
dataset_converted_path = Path(face_path / "dataset_converted").expanduser().resolve()
# распознование лиц
face_model_path = Path(model_path / "face" / "face_classifier.pkl").expanduser().resolve()
face_model_old_path = Path(model_path / "old").expanduser().resolve()
# распознование текста из звука
vosk_model_path = Path(model_path / "voice" / "vosk-model-small-ru-0.22").expanduser().resolve()
# распознованите действий из звука
yamnet_model_path = Path(model_path / "voice" / "yamnet").expanduser().resolve()

# логирование
log_path = Path(data_path / "log").expanduser().resolve()
# распознование лиц
face_attendance_path = Path(log_path / "face_attendance.csv").expanduser().resolve()
face_attendance_old_path = Path(log_path / "face_old").expanduser().resolve()
# распознование текста из звука
vosk_world_path = Path(log_path / "vosk_world.csv").expanduser().resolve()
vosk_world_old_path = Path(log_path / "vosk_old").expanduser().resolve()
