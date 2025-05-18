import tensorflow_hub as hub
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
