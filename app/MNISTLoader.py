import numpy as np
from datasets import load_dataset
from PIL import Image
from PyQt6.QtWidgets import QMessageBox

def binarize_image(image):
	"""Binaryzuje obraz do wartości {-1, 1}"""
	return np.array(image.convert("1"), dtype=int) * 2 - 1

def resize_image(image, target_size):
	"""Zmienia rozmiar obrazu używając PIL."""
	#if isinstance(image, np.ndarray):
	#	image = Image.fromarray(image.astype(np.uint8))
	return image.resize(target_size, Image.Resampling.NEAREST)

def load_fashion_mnist_patterns(parent, num_patterns=5, target_size=(28, 28)):
	"""Ładuje wzorce z MNIST Fashion używając datasets"""
	try:
		# Załaduj dataset
		dataset = load_dataset("fashion_mnist", split="test")
		
		images = dataset[:num_patterns]['image']
		patterns = []
		
		# Iteruj przez dataset i zbierz wzorce
		for img in images:
			resized = resize_image(img, target_size)
			binary_pattern = binarize_image(resized)

			patterns.append(binary_pattern)

		return patterns
		
	except Exception as e:
		if parent:
			QMessageBox.critical(parent, "Błąd", f"Błąd podczas ładowania MNIST Fashion: {str(e)}")
		return None

def get_random_fashion_mnist_image(parent, num_patterns=5, target_size=(28, 28)):
	"""Pobiera losowy obrazek z MNIST Fashion"""
	try:
		# Załaduj dataset
		dataset = load_dataset("fashion_mnist", split="test")
		
		# Wybierz losowy obrazek
		random_idx = np.random.randint(0, num_patterns)
		image = dataset[random_idx]['image']
		
		# Zmień rozmiar i binaryzuj
		resized = resize_image(image, target_size)
		binary_pattern = binarize_image(resized)
		
		return binary_pattern
		
	except Exception as e:
		QMessageBox.critical(parent, "Błąd", f"Błąd podczas ładowania obrazka: {str(e)}")
		return None

if __name__ == "__main__":
	import app
	app.main()