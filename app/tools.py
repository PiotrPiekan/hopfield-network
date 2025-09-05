import numpy as np

#	Chciałem sprawdzić czy wzorce na pewno się od siebie dostatecznie różnią.
#	Dobrze by było dodać to do aplikacji, kiedyś.

def calculate_hamming_distances(patterns):
	"""
	Oblicza odległości Hamminga między wszystkimi parami wzorców.
	Odległości znormalizowane do zakresu (0 - 1)
	
	Parameters
	----------
	patterns : list of ndarray
		Lista wzorców binarnych {-1, 1}
		
	Returns
	-------
	ndarray
		Macierz odległości Hamminga
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

if __name__ == "__main__":
	import MNISTLoader

	patterns = MNISTLoader.load_fashion_mnist_patterns(parent=None, num_patterns=5, target_size=(28, 28))

	print("HAMM")
	print(calculate_hamming_distances(patterns))