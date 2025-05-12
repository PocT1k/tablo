from pathlib import Path

data_path = Path(__file__).parent.parent / "data"
setting_json_path = data_path / "setting.json"
dataset_path = data_path / "face"
model_path = data_path / "model"
face_path = model_path / "face_classifier.pkl"
