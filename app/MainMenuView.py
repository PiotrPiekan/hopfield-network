from PyQt6.QtWidgets import (QVBoxLayout, QHBoxLayout, QPushButton, QLabel)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

import ModelIO
from BaseView import BaseView

class MainMenuView(BaseView):
	"""Główny widok menu aplikacji"""
	
	def __init__(self, main_window):
		"""Inicjalizuje widok głównego menu"""
		super().__init__(main_window)
		self.setup_ui()
	
	def setup_ui(self):
		"""Tworzy interfejs użytkownika głównego menu"""
		layout = QVBoxLayout()
		layout.setContentsMargins(100, 80, 100, 80)
		layout.setSpacing(40)
		layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
		
		# Tytuł aplikacji
		title_label = QLabel("Demonstracja sieci Hopfielda")
		title_label.setFont(QFont("Segoe UI", 36, QFont.Weight.Bold))
		title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		layout.addWidget(title_label)
		
		# Opis
		desc_label = QLabel("Wybierz opcję aby rozpocząć:")
		desc_label.setFont(QFont("Segoe UI", 18))
		desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		layout.addWidget(desc_label)
		
		# Przyciski główne
		button_layout = QHBoxLayout()
		button_layout.setSpacing(60)
		button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
		
		create_button = QPushButton("Stwórz nowe wzorce")
		create_button.setFont(QFont("Segoe UI", 20))
		create_button.setStyleSheet("""padding: 0.75em 1.5em;""")
		create_button.clicked.connect(self.switch_to_pattern_edit)
		button_layout.addWidget(create_button)
		
		import_button = QPushButton("Importuj gotowy model")
		import_button.setFont(QFont("Segoe UI", 20))
		import_button.setStyleSheet("""padding: 0.75em 1.5em;""")
		import_button.clicked.connect(self.import_model)
		button_layout.addWidget(import_button)
		
		layout.addLayout(button_layout)
		
		# Informacje o modelu
		info_label = QLabel("Model Hopfielda służy do odtwarzania zapamiętanych wzorców z zaszumionych danych.\nDemonstracja na przykładzie obrazków binarnych.")
		info_label.setFont(QFont("Segoe UI", 14))
		info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		layout.addWidget(info_label)
		
		layout.addStretch()
		
		self.setLayout(layout)
	
	def switch_to_pattern_edit(self):
		"""Przechodzi do widoku edycji wzorców"""
		self.main_window.switch_to_view("PatternEdit")
	
	def import_model(self):
		"""Importuje model z pliku i przechodzi do testowania"""
		model_data = ModelIO.import_model(self)
		if model_data:
			self.main_window.switch_to_view("ModelTest", model_data)

if __name__ == "__main__":
	import app
	app.main()