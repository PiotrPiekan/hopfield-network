import numpy as np
import pickle
import os

#	Funkcje pomocnicze do raportu, których nie chciało mi się poprawnie integrować z aplikacją.

def calculate_hamming_distances(patterns):
	"""
	Oblicza odległości Hamminga między wszystkimi parami wzorców.
	Odległości znormalizowane do zakresu (0 - 1)
	"""
	if not patterns:
		return None
		
	n_patterns = len(patterns)
	pattern_size = patterns[0].size
	
	# Spłaszcz wzorce do wektorów
	patterns = [p.flatten() for p in patterns]
	
	# Macierz odległości
	distances = np.zeros((n_patterns, n_patterns))
	
	for i in range(n_patterns):
		for j in range(n_patterns):
			if i == j:
				distances[i, j] = 0.0
			else:
				distance_in_bits = (pattern_size - np.dot(patterns[i], patterns[j])) / 2
				distances[i, j] = distance_in_bits / pattern_size
	
	return distances

def save_images_to_file(input_pattern, output_pattern, filename="recall_results.pkl"):
	"""
	Zapisuje obrazek wejściowy i wyjściowy do pliku w formacie pickle.
	"""
	try:
		# Sprawdź czy plik już istnieje
		if os.path.exists(filename):
			# Wczytaj istniejące dane
			with open(filename, 'rb') as f:
				existing_data = pickle.load(f)
			input_images = existing_data['input_images']
			output_images = existing_data['output_images']
		else:
			# Utwórz nowe listy
			input_images = []
			output_images = []
		
		# Dodaj nowe obrazki
		input_images.append(input_pattern)
		output_images.append(output_pattern)
		
		# Przygotuj dane do zapisu
		data_to_save = {
			'input_images': input_images,
			'output_images': output_images
		}
		
		# Zapisz zaktualizowane dane
		with open(filename, 'wb') as f:
			pickle.dump(data_to_save, f)
		
		return True
		
	except Exception as e:
		print(f"Nie udało się zapisać obrazków:\n{str(e)}")
		return False

def load_and_display_saved_images(filename="recall_results.pkl"):
	"""
	Funkcja pomocnicza do wczytywania i wyświetlania zapisanych obrazków w matplotlib.
	"""
	import matplotlib.pyplot as plt
	
	try:
		with open(filename, 'rb') as f:
			data = pickle.load(f)
		
		input_images = data['input_images']
		output_images = data['output_images']
		
		n_images = len(input_images)
		
		accuracies = []
		titles = [(f"Obraz {i + 1}", f"Poprawność {accuracies[i]}") for i in len(n_images)]
		
		# Utwórz subplot dla wszystkich par obrazków
		fig, axes = plt.subplots(n_images, 2, figsize=(10, 2.5*n_images))
		if n_images == 1:
			axes = axes.reshape(1, -1)
		
		for i in range(n_images):
			# Obrazek wejściowy
			axes[i, 0].imshow(input_images[i], cmap='gray', vmin=-1, vmax=1)
			axes[i, 0].set_title(f"Wejście {i + 1}")
			axes[i, 0].set_xticks([])
			axes[i, 0].set_yticks([])
			
			# Obrazek wyjściowy
			axes[i, 1].imshow(output_images[i], cmap='gray', vmin=-1, vmax=1)
			axes[i, 1].set_title(f"Wyjście {i + 1} - {accuracies[i]:.1f}%")
			axes[i, 1].set_xticks([])
			axes[i, 1].set_yticks([])
		
		plt.tight_layout()
		plt.show()
		
		print(f"Wczytano {n_images} par obrazków z pliku {filename}")
		
	except Exception as e:
		print(f"Błąd podczas wczytywania pliku: {str(e)}")

if __name__ == "__main__":
	import sys
	import os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

	# Następnie import z aplikacji:
	import MNISTLoader

	patterns = MNISTLoader.load_fashion_mnist_patterns(parent=None, num_patterns=5, target_size=(28, 28))

	print("HAMM")
	print(calculate_hamming_distances(patterns))