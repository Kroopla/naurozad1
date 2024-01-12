import sys
from PyQt5 import QtWidgets, QtGui, QtCore
import neural_network


class ButtonGrid(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Główny layout pionowy
        main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(main_layout)

        # Layout siatki dla przycisków
        grid_layout = QtWidgets.QGridLayout()

        self.buttons = []  # Lista do przechowywania przycisków
        self.buttonState = []

        # Tworzenie przycisków i dodawanie ich do siatki
        for row in range(7):
            for col in range(5):
                button = QtWidgets.QPushButton()  # Tworzenie nowego przycisku
                button.setFixedSize(50, 50)  # Ustawienie stałego rozmiaru dla przycisków
                button.setStyleSheet("background-color: white")  # Ustawienie początkowego koloru na biały

                # Dodanie przycisku do siatki
                grid_layout.addWidget(button, row, col)

                # Połączenie przycisku ze slotem zmieniającym jego kolor
                button.clicked.connect(self.change_color)
                button.clicked.connect(lambda: neural_network.test(self.buttonState))

                # Dodanie przycisku do listy
                self.buttons.append(button)
                self.buttonState.append(0)

        # Dodanie layoutu siatki do głównego layoutu
        main_layout.addLayout(grid_layout)

        # Pole tekstowe do wyświetlania wyników
        self.result_text = QtWidgets.QLabel("")
        self.result_text.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(self.result_text)

        # Layout dla przycisków poniżej siatki
        buttons_layout = QtWidgets.QHBoxLayout()

        # Tworzenie przycisku "Trenuj"
        train_button = QtWidgets.QPushButton('Trenuj')
        train_button.clicked.connect(lambda: neural_network.trening())
        buttons_layout.addWidget(train_button)

        # Tworzenie przycisku "Wykrywaj"
        detect_button = QtWidgets.QPushButton('Wykrywaj')
        detect_button.clicked.connect(lambda: neural_network.test(self.buttonState))
        buttons_layout.addWidget(detect_button)

        # Tworzenie przycisku "Wyczyść"
        clear_button = QtWidgets.QPushButton('Wyczyść')
        clear_button.clicked.connect(self.clear_grid)  # Połączenie przycisku z funkcją czyszczącą siatkę
        buttons_layout.addWidget(clear_button)

        # Dodanie layoutu przycisków do głównego layoutu
        main_layout.addLayout(buttons_layout)

        # Ustawienia okna
        self.setWindowTitle('Zadanie 1')

    def change_color(self):
        # Zmiana koloru przycisku po kliknięciu
        button = self.sender()  # Otrzymujemy przycisk, który został kliknięty
        color = button.palette().button().color()  # Pobieramy aktualny kolor przycisku
        index = self.buttons.index(button)

        # Sprawdzanie aktualnego koloru i zmiana na przeciwny
        if color == QtGui.QColor('white'):
            button.setStyleSheet("background-color: red")
            self.buttonState[index] = 1
        else:
            button.setStyleSheet("background-color: white")
            self.buttonState[index] = 0

        print(self.buttonState)

    def clear_grid(self):
        # Zresetowanie koloru wszystkich przycisków w siatce
        for button in self.buttons:
            button.setStyleSheet("background-color: white")

        for x in range(0, 35):
            self.buttonState[x] = 0


def run():
    app = QtWidgets.QApplication(sys.argv)
    window = ButtonGrid()
    window.show()
    sys.exit(app.exec_())