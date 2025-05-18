from ultralytics import YOLO
import cv2
import json
from conf import SETTING_JSON_PATH, YOLO_MODEL_PATH, YOLO_CLASSEC_PATH
from load import dict_get_or_set, write_attendance_dated

with open(SETTING_JSON_PATH, "r", encoding="utf-8") as f:
    settings = json.load(f)

yolo_model = YOLO(YOLO_MODEL_PATH)

yolo_indices = dict_get_or_set(settings, "yolo_classes", {})
confidence_thr = dict_get_or_set(settings, "yolo_threshold", 0.5)

# 3) Инициализируем видео и файл для логов
cap = cv2.VideoCapture(0)

all_indices = set()
for grp, inds in yolo_indices.items():
    all_indices.update(inds)

# — собрать обратный маппинг: индекс → список групп
index_to_groups = {}
for grp, inds in yolo_indices.items():
    for idx in inds:
        index_to_groups.setdefault(idx, []).append(grp)

def proc_yolo(frame):
    # при обработке кадра
    results = yolo_model(frame, verbose=False)[0]

    detected = []
    for cls, conf in zip(results.boxes.cls, results.boxes.conf):
        idx = int(cls)
        if conf < confidence_thr:
            continue
        for grp in index_to_groups.get(idx, []):
            if grp not in detected:
                detected.append(grp)

    if not detected:
        detected = ['None']

    # Логируем строку через ту же функцию, что и YamNet
    write_attendance_dated(
        YOLO_CLASSEC_PATH,
        detected,
        None,
        "[YOLO]"
    )

    # отрисовка
    for cls, conf, box in zip(results.boxes.cls, results.boxes.conf, results.boxes.xyxy):
        idx = int(cls)
        if idx in all_indices and conf >= confidence_thr:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame, f"{results.names[idx]} {conf:.2f}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 0, 0), 2
            )

    # показ
    cv2.imshow('Items', frame)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    proc_yolo(frame)

    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
