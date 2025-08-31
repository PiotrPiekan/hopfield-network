import sys
from PyQt6.QtWidgets import QApplication
from MainWindow import MainWindow

def main():
	"""Główna funkcja uruchamiająca aplikację"""
	app = QApplication(sys.argv)
	app.setStyle('Fusion')

	window = MainWindow()
	window.show()

	sys.exit(app.exec())

if __name__ == "__main__":
	main()