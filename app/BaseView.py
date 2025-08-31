from PyQt6.QtWidgets import QWidget

class BaseView(QWidget):
	"""Klasa bazowa dla wszystkich widoków"""
	
	def __init__(self, main_window):
		"""Inicjalizuje widok bazowy"""
		super().__init__(main_window)
		self.main_window = main_window

if __name__ == "__main__":
	import app
	app.main()