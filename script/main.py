import sys, json
from PyQt5.QtWidgets import QApplication

from conf import data_path, setting_json_path
from load import check_exist, load_face_model
from funk import dict_get_or_set
from ui import MainWindow


def main():
    check_exist(setting_json_path)
    with open(setting_json_path, "r", encoding="utf-8") as file:
        setting_json = json.load(file)
    face_model = load_face_model(setting_json)

    app = QApplication(sys.argv)
    window = MainWindow(setting_json, face_model)
    window.show()

    sys.exit(app.exec_())  # цикл событий


if __name__ == "__main__":
    main()
