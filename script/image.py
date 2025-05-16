import os
import cv2


def convert_image(dataset_to_convert_path, dataset_converted_path):
    os.makedirs(dataset_converted_path, exist_ok=True)

    for student_name in os.listdir(dataset_to_convert_path):
        student_dir = os.path.join(dataset_to_convert_path, student_name)
        output_dir = os.path.join(dataset_converted_path, student_name)
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.isdir(student_dir):
            continue

        for img_name in os.listdir(student_dir):
            img_path = os.path.join(student_dir, img_name)

            # Пропускаем не-изображения
            valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
            if not img_name.lower().endswith(valid_extensions):
                print(f"[SKIP] Пропущен не-изображение: {img_path}")
                continue

            image = cv2.imread(img_path)
            if image is None:
                print(f"[WARNING] Не удалось загрузить {img_path}")
                continue

            try:
                # image = cv2.resize(image, (256, 256)) # ресайз

                # Улучшение контраста
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray_norm = clahe.apply(gray)

                # После улучшения контраста GRAY → RGB
                rgb_image = cv2.cvtColor(gray_norm, cv2.COLOR_GRAY2RGB)

                # Сохранение
                output_file = os.path.join(output_dir, os.path.splitext(img_name)[0] + ".jpg")
                cv2.imwrite(output_file, rgb_image)  # автоматически как RGB

                print(f"[INFO] Сохранено {output_file}")

            except Exception as e:
                print(f"[ERROR] Ошибка при обработке {img_path}: {str(e)}")
