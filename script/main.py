#import tensorflow_hub as hub # НЕ УБИРАТЬ ЭТУ СТРОЧКУ БЛЯТЬ!!! ЭТА ШТУКА ДОЛЖНА ПОДКЛЮЧАТЬСЯ В ФАЙЛЕ ТОЧКИ ВХОДА!!!
import sys
from PyQt5.QtWidgets import QApplication

from ui import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())  # цикл событий


if __name__ == "__main__":
    main()

#TODO вывод действия yamnet
