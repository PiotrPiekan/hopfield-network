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

def save_images_to_file(input_pattern, output_pattern, filename="noisy_recall_results.pkl"):
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

def load_and_display_saved_images(filename="./data/recall_results.pkl"):
	"""
	Funkcja pomocnicza do wczytywania i wyświetlania zapisanych obrazków w matplotlib.
	"""
	import matplotlib.pyplot as plt
	import matplotlib.gridspec as gridspec
	
	with open(filename, 'rb') as f:
		data = pickle.load(f)
	
	input_images = data['input_images']
	output_images = data['output_images']
	
	n_pairs = len(input_images)

	accuracies = [72.6, 100.0, 92.0, 96.9, 95.9]
	#accuracies = [72.6, 100.0, 92.0, 96.9, 95.9]
	titles = [(f"Obraz {i + 1}", f"Poprawność: {accuracies[i]}%") for i in range(n_pairs)]
	#titles = [(f"Obraz {i + 1} (20% szumu)", f"Poprawność: {accuracies[i]}%") for i in range(n_pairs)]
	
	# Utwórz subplot dla wszystkich par obrazków
	n_rows = (n_pairs + 1) // 2
	fig = plt.figure(figsize = (12, 3 * n_rows))
	gs = gridspec.GridSpec(n_rows, 2, figure=fig, width_ratios=[1,1],
												wspace=0.1, hspace=0.3)
	
	for i in range(n_pairs):
		col = 0 if i % 2 == 0 else 1
		row = i // 2

		# Obrazek wejściowy
		gs_pair = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[row, col], hspace=0, wspace=-0.05)
		ax1 = fig.add_subplot(gs_pair[0])
		ax2 = fig.add_subplot(gs_pair[1])
	
		ax1.imshow(input_images[i].reshape(28, 28), cmap='gray', vmin=-1, vmax=1)
		ax1.set_title(titles[i][0], fontsize=16)
		ax1.set_xticks([])
		ax1.set_yticks([])
		
		# Obrazek wyjściowy
		ax2.imshow(output_images[i].reshape(28, 28), cmap='gray', vmin=-1, vmax=1)
		ax2.set_title(titles[i][1], fontsize=16)
		ax2.set_xticks([])
		ax2.set_yticks([])

	#plt.tight_layout()
	plt.subplots_adjust(top=0.95, bottom=0.03, left=0, right=1)
	plt.show()
	
	print(f"Wczytano {n_pairs} par obrazków z pliku {filename}")

def print_hamming_distances():
	import sys
	import os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

	# Następnie import z aplikacji:
	import MNISTLoader

	patterns = MNISTLoader.load_fashion_mnist_patterns(parent=None, num_patterns=5, target_size=(28, 28))

	print(calculate_hamming_distances(patterns))

if __name__ == "__main__":
	#load_and_display_saved_images()
	print_hamming_distances()