import tensorflow_hub as hub
import sys
from PyQt5.QtWidgets import QApplication

from ui import MainWindow


def main():
    #
    # from load import write_attendance_dated
    # write_attendance_dated("C:\Users\novik\PycharmProjects\tablo\data\log\123.txt")

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())  # цикл событий


if __name__ == "__main__":
    main()
