from pathlib import Path

data_path = Path(__file__).parent.parent / "data"
setting_json_path = data_path / "setting.json"
face_path = data_path / "face"
dataset_to_convert_path = face_path / "dataset"
dataset_converted_path = face_path / "dataset_converted"

model_path = data_path / "model"
face_model_path = model_path / "face_classifier.pkl"
face_model_old_path = model_path / "old"

log_path = data_path / "log"
attendance_path = log_path / "attendance.csv"
attendance_old_path = log_path / "old"
