import os
import cv2
import numpy as np
import face_recognition
from sklearn.svm import SVC
import pickle
from datetime import datetime

from conf import face_path


model_path = face_path


def mark_attendance(name):
    with open("attendance.csv", "a") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{name},{now}\n")
    print(f"[ATTENDANCE] {name} отмечен(а) в {now}")


print("[INFO] Загружаем обученную модель...")
with open(model_path, "rb") as f:
    model = pickle.load(f)

print("[INFO] Запуск видеопотока...")
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[ERROR] Не удалось захватить кадр с веб-камеры.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    rgb_small_frame = np.array(rgb_small_frame, dtype=np.uint8)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        predictions = model.predict_proba([face_encoding])[0]
        max_index = np.argmax(predictions)
        name = model.classes_[max_index]
        confidence = predictions[max_index]

        if confidence < 0.6:
            name = "Unknown"

        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({confidence * 100:.1f}%)", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        if name != "Unknown":
            mark_attendance(name)

    cv2.imshow('Система учета посещаемости', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
