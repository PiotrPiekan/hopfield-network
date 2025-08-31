import pickle
from os.path import splitext
from PyQt6.QtWidgets import QFileDialog, QMessageBox

def export_model(parent, patterns, model):
	"""Eksportuje model z wzorcami, wagami i biasami do pliku"""
	if not patterns or not model:
		QMessageBox.warning(parent, "Błąd", "Brak wzorców lub modelu do eksportu")
		return False
	
	try:
		# Wybierz plik do zapisu
		file_path, _ = QFileDialog.getSaveFileName(
			parent,
			"Zapisz model",
			"",
			"Pickle files (*.pkl);;All files (*.*)"
		)
		
		if not file_path:
			return False
		
		# Dodaj rozszerzenie .pkl jeśli nie ma żadnego rozszerzenia
		if not splitext(file_path)[1]:
			file_path += '.pkl'
		
		# Przygotuj dane do zapisu
		model_data = {
			'patterns': patterns,
			'weights': model.weights,
			'biases': model.biases
		}
		
		# Zapisz do pliku
		with open(file_path, 'wb') as f:
			pickle.dump(model_data, f)
		
		QMessageBox.information(parent, "Sukces", "Model został zapisany pomyślnie")
		return True
			
	except Exception as e:
		QMessageBox.critical(parent, "Błąd", f"Nie udało się zapisać modelu:\n{str(e)}")
		return False

def import_model(parent):
	"""Importuje model z pliku"""
	try:
		# Wybierz plik do wczytania
		file_path, _ = QFileDialog.getOpenFileName(
			parent,
			"Wybierz plik modelu",
			"",
			"Pickle files (*.pkl);;All files (*.*)"
		)
		
		if not file_path:
			return None
		
		# Wczytaj dane z pliku
		with open(file_path, 'rb') as f:
			model_data = pickle.load(f)
		
		# Walidacja struktury danych
		required_keys = ['patterns', 'weights', 'biases']
		if not all(key in model_data for key in required_keys):
			QMessageBox.critical(parent, "Błąd", "Niepoprawna struktura pliku modelu")
			return None
		
		QMessageBox.information(parent, "Sukces", "Model został wczytany pomyślnie")
		return model_data
		
	except Exception as e:
		QMessageBox.critical(parent, "Błąd", f"Nie udało się wczytać modelu:\n{str(e)}")
		return None

if __name__ == "__main__":
	import app
	app.main()