import os
import pickle

from conf import face_path


def check_exist(file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

def load_face_model(setting_json):
    if os.path.exists(face_path):
        with open(face_path, "rb") as f:
            face_model = pickle.load(f)
        print("[INFO] Модель распознования лиц загружена")
    else:
        face_model = None
        print("[INFO] Модель распознования не обнаружена")

    return face_model
