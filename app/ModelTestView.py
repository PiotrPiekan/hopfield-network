import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (QCheckBox, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
														 QFrame, QSpinBox, QMessageBox, QTabWidget, QWidget, QSlider,
														 QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

import ModelIO
from BaseView import BaseView
from PixelGridCanvas import PixelGridCanvas
from Hopfield import Hopfield


import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))
from tools import save_images_to_file

class ModelTestView(BaseView):
	"""Widok do testowania wytrenowanego modelu sieci Hopfielda"""
	
	def __init__(self, main_window):
		"""Inicjalizuje widok testowania modelu"""
		super().__init__(main_window)
		self.patterns = []
		self.model = None
		self.input_canvas = None
		self.output_canvas = None
		self.energy_plot = None
		self.grid_width = 28
		self.grid_height = 28
		self.recall_history = []
		self.similarity_table = None
		self.setup_ui()
		
	def setup_ui(self):
		"""Tworzy interfejs użytkownika widoku testowania"""
		main_layout = QHBoxLayout()
		main_layout.setContentsMargins(20, 20, 20, 20)
		
		# Lewy panel z opcjami
		left_panel = self.create_left_panel()
		main_layout.addWidget(left_panel)
		
		# Środkowy panel z wejściem
		middle_panel = self.create_middle_panel()
		main_layout.addWidget(middle_panel, 4)
		
		# Prawy panel z wynikami
		right_panel = self.create_right_panel()
		main_layout.addWidget(right_panel, 4)
		
		self.setLayout(main_layout)
		self.create_canvases()
	
	def create_left_panel(self):
		"""Tworzy lewy panel z opcjami i przyciskami sterującymi"""
		panel = QFrame()
		panel.setFrameStyle(QFrame.Shape.StyledPanel)
		layout = QVBoxLayout(panel)
		layout.setContentsMargins(15, 15, 15, 15)
		layout.setSpacing(15)
		
		# Tytuł sekcji opcji
		title_label = QLabel("Opcje")
		title_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
		title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		layout.addWidget(title_label)
		
		# Przyciski zarządzania modelami
		edit_button = QPushButton("Edytuj wzorce")
		edit_button.setFont(QFont("Segoe UI", 12))
		edit_button.setMinimumHeight(25)
		edit_button.clicked.connect(self.switch_to_pattern_edit)
		layout.addWidget(edit_button)
		
		import_button = QPushButton("Importuj model")
		import_button.setFont(QFont("Segoe UI", 12))
		import_button.setMinimumHeight(25)
		import_button.clicked.connect(self.import_model)
		layout.addWidget(import_button)
		
		export_button = QPushButton("Eksportuj model")
		export_button.setFont(QFont("Segoe UI", 12))
		export_button.setMinimumHeight(25)
		export_button.clicked.connect(self.export_model)
		layout.addWidget(export_button)
		
		layout.addStretch()

		# Separator
		separator = QFrame()
		separator.setFrameShape(QFrame.Shape.HLine)
		separator.setFrameShadow(QFrame.Shadow.Sunken)
		layout.addWidget(separator)
		
		layout.addStretch()
		
		# Przyciski zastąpienia całego wzorca wejściowego kolorem
		fill_white_button = QPushButton("Wypełnij białym")
		fill_white_button.setFont(QFont("Segoe UI", 12))
		fill_white_button.setMinimumHeight(25)
		fill_white_button.clicked.connect(self.fill_white)
		layout.addWidget(fill_white_button)
		
		fill_black_button = QPushButton("Wypełnij czarnym")
		fill_black_button.setFont(QFont("Segoe UI", 12))
		fill_black_button.setMinimumHeight(25)
		fill_black_button.clicked.connect(self.fill_black)
		layout.addWidget(fill_black_button)

		# Lista wzorców treningowych do wyboru
		pattern_layout = QHBoxLayout()
		
		pattern_label = QLabel("Wzorzec:")
		pattern_label.setFont(QFont("Segoe UI", 12))
		pattern_layout.addWidget(pattern_label)
		
		self.pattern_spinbox = QSpinBox()
		self.pattern_spinbox.setRange(1, 1)
		self.pattern_spinbox.setValue(1)
		self.pattern_spinbox.setFont(QFont("Segoe UI", 12))
		self.pattern_spinbox.setMinimumHeight(25)
		pattern_layout.addWidget(self.pattern_spinbox)
		
		layout.addLayout(pattern_layout)
		
		load_pattern_button = QPushButton("Załaduj wzorzec")
		load_pattern_button.setFont(QFont("Segoe UI", 12))
		load_pattern_button.setMinimumHeight(25)
		load_pattern_button.clicked.connect(self.load_selected_pattern)
		layout.addWidget(load_pattern_button)
		
		layout.addStretch()
		
		# Kontrolki szumu
		noise_layout = QHBoxLayout()
		
		noise_label = QLabel("Szum (%):")
		noise_label.setFont(QFont("Segoe UI", 12))
		noise_layout.addWidget(noise_label)
		
		self.noise_spinbox = QSpinBox()
		self.noise_spinbox.setRange(0, 100)
		self.noise_spinbox.setValue(20)
		self.noise_spinbox.setFont(QFont("Segoe UI", 12))
		self.noise_spinbox.setMinimumHeight(25)
		noise_layout.addWidget(self.noise_spinbox)
		
		layout.addLayout(noise_layout)

		add_noise_button = QPushButton("Dodaj szum")
		add_noise_button.setFont(QFont("Segoe UI", 12))
		add_noise_button.setMinimumHeight(25)
		add_noise_button.clicked.connect(self.add_noise)
		layout.addWidget(add_noise_button)
		
		# Opcje algorytmu
		max_iterations_layout = QHBoxLayout()
		
		max_iterations_label = QLabel("Liczba iteracji:")
		max_iterations_label.setFont(QFont("Segoe UI", 12))
		max_iterations_layout.addWidget(max_iterations_label)
		
		self.max_iterations_spinbox = QSpinBox()
		self.max_iterations_spinbox.setRange(1, 1000)
		self.max_iterations_spinbox.setValue(10)
		self.max_iterations_spinbox.setFont(QFont("Segoe UI", 12))
		self.max_iterations_spinbox.setMinimumHeight(25)
		max_iterations_layout.addWidget(self.max_iterations_spinbox)
		
		layout.addLayout(max_iterations_layout)

		self.early_stopping_checkbox = QCheckBox("Wczesne zatrzymanie")
		self.early_stopping_checkbox.setFont(QFont("Segoe UI", 12))
		self.early_stopping_checkbox.setChecked(True)
		layout.addWidget(self.early_stopping_checkbox)

		self.synchronous_checkbox = QCheckBox("Odtwarzanie synchroniczne")
		self.synchronous_checkbox.setFont(QFont("Segoe UI", 12))
		self.synchronous_checkbox.setChecked(True)
		layout.addWidget(self.synchronous_checkbox)
		
		# Główny przycisk odtwarzania
		recall_button = QPushButton("Odtwórz wzorzec")
		recall_button.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
		recall_button.setMinimumHeight(30)
		recall_button.clicked.connect(self.recall_pattern)
		layout.addWidget(recall_button)
		
		layout.addStretch()
		return panel
	
	def create_middle_panel(self):
		"""Tworzy środkowy panel z wzorcem wejściowym"""
		panel = QFrame()
		panel.setFrameStyle(QFrame.Shape.StyledPanel)
		layout = QVBoxLayout(panel)
		layout.setContentsMargins(15, 15, 15, 15)
		layout.setSpacing(15)
		
		# Tytuł
		title_label = QLabel("Wzorzec wejściowy")
		title_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
		title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		layout.addWidget(title_label)
		
		# Kontener na canvas
		canvas_widget = QWidget()
		canvas_widget.setMinimumHeight(400)
		self.input_canvas_layout = QVBoxLayout(canvas_widget)
		self.input_canvas_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
		layout.addWidget(canvas_widget)
		
		# Instrukcje obsługi
		instructions = QLabel("Lewy przycisk - czarny | Prawy przycisk - biały")
		instructions.setFont(QFont("Segoe UI", 9))
		instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
		instructions.setStyleSheet("color: #888;")
		layout.addWidget(instructions)
		
		return panel
	
	def create_right_panel(self):
		"""Tworzy prawy panel z wynikami odtwarzania"""
		panel = QFrame()
		panel.setFrameStyle(QFrame.Shape.StyledPanel)
		layout = QVBoxLayout(panel)
		layout.setContentsMargins(15, 15, 15, 15)
		layout.setSpacing(15)
		
		# Tytuł
		title_label = QLabel("Wyniki")
		title_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
		title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		layout.addWidget(title_label)
		
		# Zakładki z wynikami
		tab_widget = QTabWidget()
		tab_widget.setFont(QFont("Segoe UI", 10))
		tab_widget.setTabPosition(QTabWidget.TabPosition.South)

		# Zakładka z odtworzonym wzorcem
		canvas_tab = QWidget()
		canvas_tab_layout = QVBoxLayout(canvas_tab)
		canvas_tab_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

		# Kontener na canvas wyjściowy
		canvas_container = QWidget()
		canvas_container.setMinimumHeight(400)
		self.canvas_container_layout = QVBoxLayout(canvas_container)
		self.canvas_container_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
		canvas_tab_layout.addWidget(canvas_container)
		
		# Kontrolki suwaka iteracji
		iteration_controls = QHBoxLayout()
		
		iteration_label = QLabel("Iteracja:")
		iteration_label.setFont(QFont("Segoe UI", 10))
		iteration_controls.addWidget(iteration_label)
		
		self.iteration_slider = QSlider(Qt.Orientation.Horizontal)
		self.iteration_slider.setMinimum(0)
		self.iteration_slider.setMaximum(0)
		self.iteration_slider.setValue(0)
		self.iteration_slider.valueChanged.connect(self.on_iteration_changed)
		iteration_controls.addWidget(self.iteration_slider)
		
		self.iteration_value_label = QLabel("0 / 0")
		self.iteration_value_label.setFont(QFont("Segoe UI", 10))
		self.iteration_value_label.setMinimumWidth(50)
		iteration_controls.addWidget(self.iteration_value_label)
		
		canvas_tab_layout.addLayout(iteration_controls)
		
		tab_widget.addTab(canvas_tab, "Odtworzony wzorzec")
		
		# Zakładka z wykresem energii
		plot_tab = QWidget()
		plot_layout = QVBoxLayout(plot_tab)
		
		self.figure = Figure(figsize=(6, 4))
		self.canvas_plot = FigureCanvas(self.figure)
		plot_layout.addWidget(self.canvas_plot)
		
		tab_widget.addTab(plot_tab, "Energia")

		# Zakładka z zgodnością wzorców
		similarity_tab = QWidget()
		similarity_layout = QVBoxLayout(similarity_tab)
		
		similarity_title = QLabel("Zgodność z wzorcami treningowymi")
		similarity_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
		similarity_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
		similarity_layout.addWidget(similarity_title)
		
		self.similarity_table = QTableWidget()
		self.similarity_table.setColumnCount(2)
		self.similarity_table.setHorizontalHeaderLabels(["Wzorzec", "Zgodność (%)"])
		
		header = self.similarity_table.horizontalHeader()
		header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
		header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
		
		self.similarity_table.setAlternatingRowColors(True)
		self.similarity_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
		
		similarity_layout.addWidget(self.similarity_table)
		
		tab_widget.addTab(similarity_tab, "Zgodność wzorców")
		
		layout.addWidget(tab_widget)
		
		return panel
	
	def create_canvases(self):
		"""Tworzy canvas wejściowy i wyjściowy z odpowiednimi wymiarami"""
		if self.input_canvas:
			self.input_canvas.deleteLater()
		
		self.input_canvas = PixelGridCanvas(
			col_count=self.grid_width,
			row_count=self.grid_height,
			max_height=400,
			max_width=400,
			drawing_enabled=True,
			show_grid=True
		)
		
		self.input_canvas_layout.addWidget(self.input_canvas)
		
		if self.output_canvas:
			self.output_canvas.deleteLater()
		
		self.output_canvas = PixelGridCanvas(
			col_count=self.grid_width,
			row_count=self.grid_height,
			max_height=400,
			max_width=400,
			drawing_enabled=False,
			show_grid=True
		)
		
		self.canvas_container_layout.addWidget(self.output_canvas)
	
	def fill_white(self):
		"""Wypełnia canvas wejściowy białymi pikselami"""
		if self.input_canvas:
			self.input_canvas.fill_white()
	
	def fill_black(self):
		"""Wypełnia canvas wejściowy czarnymi pikselami"""
		if self.input_canvas:
			self.input_canvas.fill_black()
	
	def add_noise(self):
		"""Dodaje szum do wzorca wejściowego"""
		if not self.input_canvas:
			return
		
		noise_level = self.noise_spinbox.value() / 100.0
		current_pixels = self.input_canvas.get_pixels()
		
		n_flip = int(noise_level * current_pixels.size)
		if n_flip == 0:
			return
		
		flat_pixels = current_pixels.flatten()
		flip_indices = np.random.choice(len(flat_pixels), n_flip, replace=False)
		flat_pixels[flip_indices] *= -1
		
		noisy_pixels = flat_pixels.reshape(current_pixels.shape)
		self.input_canvas.set_pixels(noisy_pixels)
	
	def load_selected_pattern(self):
		"""Ładuje wybrany wzorzec do canvas wejściowego"""
		if not self.patterns:
			QMessageBox.warning(self, "Błąd", "Brak dostępnych wzorców")
			return
		
		try:
			pattern_index = self.pattern_spinbox.value() - 1
			if 0 <= pattern_index < len(self.patterns):
				selected_pattern = self.patterns[pattern_index]
				self.input_canvas.set_pixels(selected_pattern)
			else:
				QMessageBox.warning(self, "Błąd", "Nieprawidłowy numer wzorca")
		except Exception as e:
			QMessageBox.critical(self, "Błąd", f"Nie udało się załadować wzorca:\n{str(e)}")

	def update_pattern_spinbox_range(self):
			"""Aktualizuje zakres spinbox z wzorcami"""
			if self.patterns:
				self.pattern_spinbox.setRange(1, len(self.patterns))
				self.pattern_spinbox.setValue(1)
			else:
				self.pattern_spinbox.setRange(1, 1)
				self.pattern_spinbox.setValue(1)

	def recall_pattern(self):
		"""Uruchamia proces odtwarzania wzorca przez sieć"""
		if not self.input_canvas or not self.model:
			QMessageBox.warning(self, "Błąd", "Brak modelu lub danych wejściowych")
			return
		
		input_pattern = self.input_canvas.get_pixels()
		synchronous = self.synchronous_checkbox.isChecked()
		max_iterations = self.max_iterations_spinbox.value()
		early_stopping = self.early_stopping_checkbox.isChecked()
		
		try:
			states_history, energy_history = self.model.recall(
				input_pattern, 
				synchronous=synchronous,
				max_iterations=max_iterations,
				energy_tol=1e-9 if early_stopping else 0
			)
			
			self.recall_history = states_history
			
			# Zaktualizuj zakres suwaka
			self.iteration_slider.setMaximum(len(states_history) - 1)
			self.iteration_slider.setValue(len(states_history) - 1)
			
			# Pokaż końcowy wynik
			self.on_iteration_changed(len(states_history) - 1)
			
			# Narysuj wykres energii
			self.plot_energy(energy_history)

			save_images_to_file(input_pattern, states_history[-1])
			
		except Exception as e:
			QMessageBox.critical(self, "Błąd", f"Błąd podczas odtwarzania:\n{str(e)}")
	
	def on_iteration_changed(self, value):
		"""Obsługuje zmianę pozycji suwaka iteracji"""
		if self.recall_history and 0 <= value < len(self.recall_history):
			pattern = self.recall_history[value].reshape(self.grid_height, self.grid_width)
			self.output_canvas.set_pixels(pattern)
			self.update_iteration_label()
			self.update_similarity_table(pattern)
	
	def update_iteration_label(self):
		"""Aktualizuje etykietę z numerem iteracji"""
		current = self.iteration_slider.value()
		total = self.iteration_slider.maximum()
		self.iteration_value_label.setText(f"{current} / {total}")
	
	def plot_energy(self, energy_history):
		"""Rysuje wykres zmian energii podczas odtwarzania"""
		self.figure.clear()
		ax = self.figure.add_subplot(111)
		
		ax.plot(energy_history, 'b-o', markersize=4)
		ax.set_title('Zmiana energii')
		ax.set_xlabel('Iteracja')
		ax.set_ylabel('Energia')
		ax.grid(True, alpha=0.3)
		
		self.figure.tight_layout()
		self.canvas_plot.draw()

	def calculate_pattern_similarity(self, output_pattern):
		"""Oblicza zgodność wzorca wyjściowego z wzorcami treningowymi"""
		if not self.patterns:
			return []
		
		similarities = []
		
		for i, pattern in enumerate(self.patterns):
			matches = np.sum(output_pattern == pattern)
			similarity = (matches / pattern.size) * 100
			
			similarities.append((i + 1, similarity))
		
		return similarities
	
	def update_similarity_table(self, output_pattern):
		"""Aktualizuje tabelę zgodności wzorców"""
		if not self.similarity_table:
			return
		
		similarities = self.calculate_pattern_similarity(output_pattern)
		
		self.similarity_table.setRowCount(len(similarities))
		
		for row, (pattern_num, similarity) in enumerate(similarities):
			pattern_item = QTableWidgetItem(f"Wzorzec {pattern_num}")
			pattern_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
			
			similarity_item = QTableWidgetItem(f"{similarity:.1f}")
			similarity_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
			
			self.similarity_table.setItem(row, 0, pattern_item)
			self.similarity_table.setItem(row, 1, similarity_item)
	
	def set_model_data(self, patterns, existing_model=None):
		"""Ustawia dane modelu z wzorcami i opcjonalnie istniejącym modelem"""
		self.patterns = patterns
		if len(patterns) <= 0:
			return
		
		pattern_shape = patterns[0].shape
		self.grid_height, self.grid_width = pattern_shape
		
		if existing_model is not None:
			self.model = existing_model
		else:
			self.model = Hopfield(self.grid_width * self.grid_height)
			self.model.train(patterns)
		
		self.create_canvases()

	def set_patterns_and_train(self, patterns):
		"""Ustawia wzorce i trenuje nowy model"""
		self.patterns = patterns

		if len(self.patterns) <= 0:
			return
		
		self.grid_height, self.grid_width = self.patterns[0].shape
		
		self.model = Hopfield(self.grid_width * self.grid_height)
		self.model.train(patterns)
		
		self.create_canvases()
		self.update_pattern_spinbox_range()
	
	def import_model_data(self, model_data):
		"""Importuje pełne dane modelu z pliku"""
		self.patterns = model_data['patterns']
		
		if len(self.patterns) <= 0:
			return
		
		self.grid_height, self.grid_width = self.patterns[0].shape
		
		size = len(model_data['biases'])
		self.model = Hopfield(size, model_data['weights'], model_data['biases'])
		
		self.create_canvases()
		self.update_pattern_spinbox_range()
	
	def switch_to_pattern_edit(self):
		"""Przechodzi do widoku edycji wzorców"""
		self.main_window.switch_to_view("PatternEdit", self.patterns)
	
	def export_model(self):
		"""Eksportuje aktualny model do pliku"""
		ModelIO.export_model(self, self.patterns, self.model)
	
	def import_model(self):
		"""Importuje model z pliku"""
		model_data = ModelIO.import_model(self)
		if model_data:
			self.import_model_data(model_data)

if __name__ == "__main__":
	import app
	app.main()