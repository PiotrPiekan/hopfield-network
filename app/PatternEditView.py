import numpy as np
from PyQt6.QtWidgets import (QMessageBox, QHBoxLayout, QVBoxLayout, QLabel, QSpinBox, 
							QPushButton, QScrollArea, QWidget, QFrame)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from BaseView import BaseView
from PixelGridCanvas import PixelGridCanvas
import MNISTLoader

class PatternEditView(BaseView):
	"""Widok do edycji wzorców do zapamiętania przez sieć"""
	
	def __init__(self, main_window):
		"""Inicjalizuje widok edycji wzorców"""
		super().__init__(main_window)
		self.grid_width = 28
		self.grid_height = 28
		self.canvas = None

		self.current_pattern_id = -1
		self.patterns = []

		self.setup_ui()
		self.reset_patterns()
		self.create_canvas()
	
	def setup_ui(self):
		"""Tworzy interfejs użytkownika widoku edycji"""
		main_layout = QHBoxLayout()
		main_layout.setContentsMargins(20, 20, 20, 20)
		
		# Lewy panel z listą wzorców
		left_panel = self.create_left_panel()
		main_layout.addWidget(left_panel, 6)
		
		# Prawy panel z edytorem
		right_panel = self.create_right_panel()
		main_layout.addWidget(right_panel, 4)
		
		self.setLayout(main_layout)
	
	def create_left_panel(self):
		"""Tworzy lewy panel z listą wzorców i kontrolkami"""
		panel = QFrame()
		panel.setFrameStyle(QFrame.Shape.StyledPanel)
		layout = QVBoxLayout(panel)
		layout.setContentsMargins(15, 15, 15, 15)
		layout.setSpacing(10)
		
		# Górny rząd - tytuł i wymiary
		top_row = QHBoxLayout()
		
		patterns_label = QLabel("Lista wzorców")
		patterns_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
		top_row.addWidget(patterns_label)
		
		top_row.addStretch()
		
		# Kontrolki wymiarów
		dimensions_layout = QHBoxLayout()
		dimensions_layout.setSpacing(10)
		
		width_label = QLabel("Szer:")
		width_label.setFont(QFont("Segoe UI", 10))
		self.width_spinbox = QSpinBox()
		self.width_spinbox.setFont(QFont("Segoe UI", 10))
		self.width_spinbox.setRange(1, 100)
		self.width_spinbox.setValue(self.grid_width)
		
		height_label = QLabel("Wys:")
		height_label.setFont(QFont("Segoe UI", 10))
		self.height_spinbox = QSpinBox()
		self.height_spinbox.setFont(QFont("Segoe UI", 10))
		self.height_spinbox.setRange(1, 100)
		self.height_spinbox.setValue(self.grid_height)
		
		confirm_dims_button = QPushButton("OK")
		confirm_dims_button.setFont(QFont("Segoe UI", 10))
		confirm_dims_button.setMaximumWidth(50)
		confirm_dims_button.clicked.connect(self.set_pattern_dimensions)
		
		dimensions_layout.addWidget(width_label)
		dimensions_layout.addWidget(self.width_spinbox)
		dimensions_layout.addWidget(height_label)
		dimensions_layout.addWidget(self.height_spinbox)
		dimensions_layout.addWidget(confirm_dims_button)
		
		top_row.addLayout(dimensions_layout)
		layout.addLayout(top_row)
		
		# Przewijalny obszar na listę wzorców
		scroll_area = QScrollArea()
		scroll_area.setWidgetResizable(True)
		scroll_area.setMinimumHeight(300)
		
		scroll_widget = QWidget()
		self.scroll_layout = QVBoxLayout(scroll_widget)
		
		scroll_area.setWidget(scroll_widget)
		layout.addWidget(scroll_area)
		
		# Dolny rząd - przyciski akcji
		bottom_row = QHBoxLayout()
		
		cancel_button = QPushButton("Anuluj")
		cancel_button.setFont(QFont("Segoe UI", 12))
		cancel_button.setStyleSheet("""padding: 0.75em 1.5em;""")
		cancel_button.clicked.connect(self.go_back)
		bottom_row.addWidget(cancel_button)
		
		bottom_row.addStretch()
		
		# Importu z MNIST Fashion
		mnist_count_label = QLabel("Liczba wzorców:")
		mnist_count_label.setFont(QFont("Segoe UI", 12))
		bottom_row.addWidget(mnist_count_label)

		self.mnist_count_spinbox = QSpinBox()
		self.mnist_count_spinbox.setFont(QFont("Segoe UI", 12))
		self.mnist_count_spinbox.setRange(1, 100)
		self.mnist_count_spinbox.setValue(5)
		bottom_row.addWidget(self.mnist_count_spinbox)

		mnist_button = QPushButton("Pobierz wzorce z sieci")
		mnist_button.setFont(QFont("Segoe UI", 12))
		mnist_button.setStyleSheet("""padding: 0.75em 1.5em;""")
		mnist_button.clicked.connect(self.import_mnist_fashion)
		bottom_row.addWidget(mnist_button)
		
		# Przycisk tworzenia modelu
		confirm_all_button = QPushButton("Stwórz model")
		confirm_all_button.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
		confirm_all_button.setStyleSheet("""padding: 0.75em 1.5em;""")
		confirm_all_button.clicked.connect(self.create_model)
		bottom_row.addWidget(confirm_all_button)
		
		layout.addLayout(bottom_row)
		
		return panel
	
	def create_right_panel(self):
		"""Tworzy prawy panel z edytorem wzorca"""
		panel = QFrame()
		panel.setFrameStyle(QFrame.Shape.StyledPanel)
		layout = QVBoxLayout(panel)
		layout.setContentsMargins(15, 15, 15, 15)
		layout.setSpacing(15)
		
		# Tytuł
		title_label = QLabel("Edytor wzorca")
		title_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
		title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		layout.addWidget(title_label)
		
		# Kontener na canvas
		canvas_widget = QWidget()
		canvas_widget.setMinimumHeight(400)
		self.canvas_layout = QVBoxLayout(canvas_widget)
		self.canvas_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
		layout.addWidget(canvas_widget)
		
		# Przyciski edycji wzorca
		buttons_layout = QHBoxLayout()
		buttons_layout.setSpacing(10)
		
		fill_white_button = QPushButton("Wypełnij białym")
		fill_white_button.setFont(QFont("Segoe UI", 10))
		fill_white_button.clicked.connect(self.fill_white)
		buttons_layout.addWidget(fill_white_button)
		
		fill_black_button = QPushButton("Wypełnij czarnym")
		fill_black_button.setFont(QFont("Segoe UI", 10))
		fill_black_button.clicked.connect(self.fill_black)
		buttons_layout.addWidget(fill_black_button)
		
		delete_button = QPushButton("Usuń")
		delete_button.setFont(QFont("Segoe UI", 10))
		delete_button.clicked.connect(self.delete_current_pattern)
		buttons_layout.addWidget(delete_button)
		
		layout.addLayout(buttons_layout)
		
		# Instrukcje obsługi
		instructions = QLabel("Lewy przycisk - czarny | Prawy przycisk - biały")
		instructions.setFont(QFont("Segoe UI", 9))
		instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
		instructions.setStyleSheet("color: #888;")
		layout.addWidget(instructions)
		
		return panel
	
	def create_canvas(self):
		"""Tworzy nowy canvas do edycji wzorców"""
		if self.canvas:
			self.canvas.deleteLater()
		
		self.canvas = PixelGridCanvas(
			col_count=self.grid_width,
			row_count=self.grid_height,
			max_height=400,
			max_width=400
		)
		
		self.canvas.pixels_changed.connect(self.on_canvas_changed)
		self.canvas_layout.addWidget(self.canvas)

	def update_pattern_scroll_list(self):
		"""Aktualizuje listę wzorców w panelu przewijania"""
		# Usuń wszystkie istniejące elementy
		while self.scroll_layout.count() > 0:
			item = self.scroll_layout.takeAt(0)
			if item.widget():
				item.widget().deleteLater()
		
		# Dodaj przyciski dla wszystkich wzorców
		for i in range(len(self.patterns)):
			pattern_button = QPushButton(f"Wzorzec {i + 1}")
			pattern_button.setFont(QFont("Segoe UI", 10))
			pattern_button.setMinimumHeight(35)
			pattern_button.clicked.connect(lambda checked, idx=i: self.select_pattern(idx))
			self.scroll_layout.addWidget(pattern_button)
		
		# Dodaj przycisk "Dodaj wzorzec" na końcu
		add_button = QPushButton("Dodaj wzorzec")
		add_button.setFont(QFont("Segoe UI", 10))
		add_button.setMinimumHeight(35)
		add_button.clicked.connect(self.add_pattern)
		self.scroll_layout.addWidget(add_button)
		
		# Dodaj stretch na końcu
		self.scroll_layout.addStretch()
	
	def create_model(self):
		"""Przechodzi do testowania modelu z aktualnymi wzorcami"""
		if not self.patterns:
			return
		
		self.main_window.switch_to_view("ModelTest", self.patterns)
	
	def set_patterns(self, patterns):
		"""Ustawia wzorce otrzymane z innego widoku"""
		if not patterns:
			return
		
		pattern_shape = patterns[0].shape
		if pattern_shape != (self.grid_height, self.grid_width):
			self.grid_height, self.grid_width = pattern_shape
			self.height_spinbox.setValue(self.grid_height)
			self.width_spinbox.setValue(self.grid_width)
			self.create_canvas()
		
		self.patterns = patterns.copy()
		self.select_pattern(0)
		self.update_pattern_scroll_list()

	def select_pattern(self, pattern_index):
		"""Wybiera wzorzec do edycji"""
		self.current_pattern_id = pattern_index
		self.load_current_pattern_to_canvas()
	
	def add_pattern(self):
		"""Dodaje nowy pusty wzorzec"""
		new_pattern = np.full((self.grid_height, self.grid_width), -1, dtype=int)
		self.patterns.append(new_pattern)
		self.select_pattern(len(self.patterns) - 1)
		self.update_pattern_scroll_list()
	
	def delete_current_pattern(self):
		"""Usuwa aktualnie wybrany wzorzec"""
		if len(self.patterns) <= 1:
			self.fill_white()
		else:
			del self.patterns[self.current_pattern_id]
			self.select_pattern(max(0, self.current_pattern_id - 1))
			self.update_pattern_scroll_list()
	
	def reset_patterns(self):
		"""Resetuje wszystkie wzorce do jednego pustego"""
		empty_pattern = np.full((self.grid_height, self.grid_width), -1, dtype=int)
		self.patterns = [empty_pattern]
		self.select_pattern(0)
		self.update_pattern_scroll_list()
	
	def set_pattern_dimensions(self):
		"""Ustawia nowe wymiary wzorców"""
		width = self.width_spinbox.value()
		height = self.height_spinbox.value()

		if width == self.grid_width and height == self.grid_height:
			return

		self.grid_width = width
		self.grid_height = height

		self.create_canvas()
		self.reset_patterns()
	
	def import_mnist_fashion(self):
		"""Importuje wzorce z MNIST Fashion"""
		try:
			pattern_count = self.mnist_count_spinbox.value()

			# Przekaż aktualne wymiary grida
			patterns = MNISTLoader.load_fashion_mnist_patterns(
				self, target_size=(self.grid_width, self.grid_height), num_patterns=pattern_count
			)
			if patterns is not None and len(patterns) > 0:
				# Ustaw wzorce
				self.set_patterns(patterns)
				
				QMessageBox.information(self, "Sukces", f"Załadowano {len(patterns)} wzorców z MNIST Fashion")
		except Exception as e:
			QMessageBox.critical(self, "Błąd", f"Nie udało się załadować MNIST Fashion:\n{str(e)}")
	
	def load_current_pattern_to_canvas(self):
		"""Ładuje aktualnie wybrany wzorzec do canvas"""
		if self.canvas and 0 <= self.current_pattern_id < len(self.patterns):
			self.canvas.set_pixels(self.patterns[self.current_pattern_id])
	
	def on_canvas_changed(self):
		"""Obsługuje zmiany w canvas i zapisuje je do wzorca"""
		if self.canvas and 0 <= self.current_pattern_id < len(self.patterns):
			self.patterns[self.current_pattern_id] = self.canvas.get_pixels()
	
	def fill_white(self):
		"""Wypełnia canvas białymi pikselami"""
		if self.canvas:
			self.canvas.fill_white()
	
	def fill_black(self):
		"""Wypełnia canvas czarnymi pikselami"""
		if self.canvas:
			self.canvas.fill_black()
	
	def go_back(self):
		"""Wraca do poprzedniego widoku"""
		self.main_window.go_back()

if __name__ == "__main__":
	import app
	app.main()