import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen

class PixelGridCanvas(QWidget):
	"""Widget do rysowania i wyświetlania wzorców binarnych"""
	
	pixels_changed = pyqtSignal()
	
	def __init__(self, col_count=20, row_count=20, max_width=400, max_height=400, drawing_enabled=True, show_grid=True):
		"""Inicjalizuje canvas do rysowania pikseli"""
		super().__init__()
		self.col_count = col_count
		self.row_count = row_count
		self.max_width = max_width
		self.max_height = max_height
		self.drawing_enabled = drawing_enabled
		self.show_grid = show_grid
		
		max_cell_width = self.max_width // self.col_count
		max_cell_height = self.max_height // self.row_count
		self.cell_size = min(max_cell_width, max_cell_height)
		
		actual_width = self.cell_size * self.col_count
		actual_height = self.cell_size * self.row_count
		
		self.setFixedSize(actual_width, actual_height)
		
		self.pixels = np.full((self.row_count, self.col_count), -1, dtype=int)
		
		self.is_drawing = False
		self.current_color = -1
	
	def paintEvent(self, event):
		"""Rysuje grid pikseli na widget"""
		painter = QPainter(self)
		painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
		
		for y in range(self.row_count):
			for x in range(self.col_count):
				color = QColor(255, 255, 255) if self.pixels[y, x] == -1 else QColor(0, 0, 0)
				painter.fillRect(
					x * self.cell_size,
					y * self.cell_size,
					self.cell_size,
					self.cell_size,
					color
				)
		
		if self.show_grid:
			painter.setPen(QPen(QColor(200, 200, 200), 1))
			
			for x in range(self.col_count + 1):
				painter.drawLine(
					x * self.cell_size, 0,
					x * self.cell_size, self.row_count * self.cell_size
				)
			
			for y in range(self.row_count + 1):
				painter.drawLine(
					0, y * self.cell_size,
					self.col_count * self.cell_size, y * self.cell_size
				)

	def mousePressEvent(self, event):
		"""Obsługuje naciśnięcie przycisku myszy"""
		if not self.drawing_enabled:
			return
		
		x, y = self.get_grid_coordinates(event.position().x(), event.position().y())
		
		if x is not None and y is not None:
			if event.button() == Qt.MouseButton.LeftButton:
				self.current_color = 1
			elif event.button() == Qt.MouseButton.RightButton:
				self.current_color = -1
			
			self.is_drawing = True
			self.set_pixel(x, y)
	
	def mouseMoveEvent(self, event):
		"""Obsługuje ruch myszy podczas rysowania"""
		if self.is_drawing:
			x, y = self.get_grid_coordinates(event.position().x(), event.position().y())
			if x is not None and y is not None:
				self.set_pixel(x, y)
	
	def mouseReleaseEvent(self, event):
		"""Obsługuje puszczenie przycisku myszy"""
		self.is_drawing = False
		self.pixels_changed.emit()
	
	def get_grid_coordinates(self, mouse_x, mouse_y):
		"""Konwertuje współrzędne myszy na współrzędne grida"""
		x = int(mouse_x // self.cell_size)
		y = int(mouse_y // self.cell_size)
		
		if 0 <= x < self.col_count and 0 <= y < self.row_count:
			return x, y
		return None, None

	def set_pixel(self, x, y):
		"""Ustawia kolor piksela na podanych współrzędnych"""
		if self.pixels[y, x] != self.current_color:
			self.pixels[y, x] = self.current_color
			self.update(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
		
	def get_pixels(self):
		"""Zwraca kopię tablicy pikseli"""
		return self.pixels.copy()
	
	def set_pixels(self, new_pixels):
		"""Ustawia nową tablicę pikseli"""
		if new_pixels.shape != (self.row_count, self.col_count):
			raise ValueError(f"Wrong array shape. Expected {(self.row_count, self.col_count)}, got {new_pixels.shape}")
		
		self.pixels = new_pixels.copy()
		self.update()
		self.pixels_changed.emit()
	
	def fill_black(self):
		"""Wypełnia wszystkie piksele czarnym kolorem"""
		self.pixels.fill(1)
		self.update()
		self.pixels_changed.emit()
	
	def fill_white(self):
		"""Wypełnia wszystkie piksele białym kolorem"""
		self.pixels.fill(-1)
		self.update()
		self.pixels_changed.emit()

if __name__ == "__main__":
	import app
	app.main()