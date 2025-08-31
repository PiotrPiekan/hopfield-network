from PyQt6.QtWidgets import QMainWindow, QStackedWidget

from MainMenuView import MainMenuView
from PatternEditView import PatternEditView
from ModelTestView import ModelTestView

class MainWindow(QMainWindow):
	"""Główne okno aplikacji zarządzające widokami"""
	
	def __init__(self):
		"""Inicjalizuje główne okno aplikacji"""
		super().__init__()
		self.setWindowTitle("Projekt 1 Piotra Piekańskiego")
		self.setFixedSize(1300, 600)

		self.current_view = None
		self.previous_view = None

		self.setup_views()

	def setup_views(self):
		"""Tworzy i konfiguruje wszystkie widoki aplikacji"""
		# Stwórz kontener na widoki
		self.stacked_widget = QStackedWidget()
		self.setCentralWidget(self.stacked_widget)

		# Stwórz widoki
		self.main_menu = MainMenuView(self)
		self.pattern_edit = PatternEditView(self)
		self.model_test = ModelTestView(self)

		# Dodaj widoki do kontenera
		self.stacked_widget.addWidget(self.main_menu)
		self.stacked_widget.addWidget(self.pattern_edit)
		self.stacked_widget.addWidget(self.model_test)

		# Przejdź do głównego menu
		self.switch_to_view("MainMenu")

	def switch_to_view(self, view_name, data=None):
		"""Przełącza między widokami z opcjonalnym transferem danych"""
		views = {
			"MainMenu": 0,
			"PatternEdit": 1,
			"ModelTest": 2
		}
		
		# Transfer danych do odpowiedniego widoku
		if view_name == "ModelTest" and data is not None:
			# Dane mogą być wzorcami (z PatternEdit) lub pełnym modelem (z pliku)
			if isinstance(data, dict) and 'weights' in data and 'biases' in data:
				# Pełny model z pliku - zawiera weights, biases, patterns, dimensions
				self.model_test.import_model_data(data)
			else:
				# Tylko wzorce z PatternEdit
				self.model_test.set_patterns_and_train(data)
		
		elif view_name == "PatternEdit" and data is not None:
			# Wzorce z ModelTest
			self.pattern_edit.set_patterns(data)
		
		# Przełącz widok
		self.stacked_widget.setCurrentIndex(views[view_name])
		
		# Zapisz poprzedni widok
		self.previous_view = self.current_view
		self.current_view = view_name

	def go_back(self):
		"""Wraca do poprzedniego widoku"""
		if self.previous_view is not None:
			self.switch_to_view(self.previous_view)

if __name__ == "__main__":
	import app
	app.main()