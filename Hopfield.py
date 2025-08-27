import numpy as np

class Hopfield():
	"""
		Implementacja sieci Hopfielda.

		Attributes:
		----------
		weights : ndarray[float, float]
			Dwuwymiarowa macierz wag połączeń neuronów.
		biases : ndarray[float]
			Jednowymiarowa macierz wyrazów wolnych.
	"""

	def __init__(self, size):
		"""
			Inicjalizacja sieci Hopfielda.

			Arguments
			----------
			size : int
				Rozmiar sieci (liczba neuronów).
		"""

		self.size = size
		self.weights = np.zeros((size, size))
		self.biases = np.zeros(size)

	def train(self, patterns, progress_callback=None):
		"""
			Uczenie sieci na podstawie listy wzorców.
			Wzorzec jest n-wymiarową tablicą numpy z wartościami 1 / -1.
			Każdy wzorzec musi mieć tę samą liczbę wartości.

			Arguments
			----------
			patterns : list of numpy.ndarray of int
				Lista wzorców do zapamiętania.
		"""

		for i in range(len(patterns)):
			pattern = patterns[i].flatten()
			self.weights += np.outer(pattern, pattern)
			self.biases += pattern

			if progress_callback != None: 
				progress_callback(i+1, len(patterns))

		np.fill_diagonal(self.weights, 0)

		self.weights /= len(patterns)
		self.biases /= len(patterns)

	def energy(self, state):
		"""
			Obliczanie energii stanu sieci neuronów.

			state : numpy.ndarray of int
				Stan sieci.

			Returns
			----------
			float
				Energia.
		"""
		
		weight_energy = np.dot(np.dot(self.weights, state).T, state)
		biases_energy = np.dot(self.biases, state)

		return -0.5 * weight_energy - biases_energy
	
	def recall_asynchronous(self, input_pattern, max_iterations=100, energy_tol=1e-9, progress_callback=None):
		"""
			Proces asynchroniczny odtwarzania jednego z zapamiętanych wzorców na podstawie wzorca wejściowego.

			Arguments
			----------
			input_pattern : numpy.ndarray
				Wzorzec wejściowy.
			max_iterations : int
				Maksymalna liczba iteracji (aktualizacji wszystkich neuronów).

			Returns
			----------
			numpy.ndarray of int
				Odtworzony wzorzec.
		"""

		state = input_pattern.flatten()
		energy_history = [self.energy(state)]

		for iteration in range(max_iterations):
			idx = np.random.permutation(self.size)

			for i in idx:
				activation = np.dot(self.weights[i], state) + self.biases[i]
				state[i] = 1 if activation >= 0 else -1

			energy_history.append(self.energy(state))

			if abs(energy_history[-1] - energy_history[-2]) < energy_tol:
				break

			if progress_callback != None:
				progress_callback(state, energy_history, iteration + 1)

		return state, energy_history, iteration + 1

	def recall_synchronous(self, input_pattern, max_iterations=100, energy_tol=1e-9, progress_callback=None):
		"""
		
		"""

		state = input_pattern.flatten()
		energy_history = [self.energy(state)]

		for iteration in range(max_iterations):
			activation = np.dot(self.weights, state) + self.biases
			state = np.where(activation >= 0, 1, -1)

			energy_history.append(self.energy(state))

			if abs(energy_history[-1] - energy_history[-2]) < energy_tol:
				break

			if progress_callback != None:
				progress_callback(state, iteration + 1, energy_history)

		return state, iteration + 1, energy_history
	
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

if __name__ == "__main__":

	import matplotlib.pyplot as plt
	def add_noise(pattern, noise_level=0.1):
		"""
		Dodaje szum do wzorca
		
		Args:
				pattern (np.array): Oryginalny wzorzec
				noise_level (float): Poziom szumu (0-1)
				
		Returns:
				np.array: Zaszumiony wzorzec
		"""
		noisy_pattern = np.copy(pattern)
		n_flip = int(noise_level * len(pattern))
		flip_indices = np.random.choice(len(pattern), n_flip, replace=False)
		noisy_pattern[flip_indices] *= -1
		return noisy_pattern

	def pattern_overlap(pattern1, pattern2):
		"""
		Oblicza podobieństwo między wzorcami
		
		Args:
				pattern1, pattern2 (np.array): Wzorce do porównania
				
		Returns:
				float: Współczynnik podobieństwa (0-1)
		"""
		return np.sum(pattern1 == pattern2) / len(pattern1)

	# Definicja wzorców (10 neuronów)
	patterns = [
		np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1]),  # Wzorzec 1
		np.array([-1, -1, 1, 1, 1, -1, -1, 1, 1, 1]),  # Wzorzec 2
		np.array([1, -1, -1, 1, 1, 1, -1, -1, -1, 1])   # Wzorzec 3
	]
	
	# Utworzenie i trening sieci
	network = Hopfield(10)
	network.train(patterns)
	
	print("=== Sieć Hopfielda - Test ===")
	print(f"Rozmiar sieci: {network.size} neuronów")
	print(f"Liczba wzorców: {len(patterns)}")
	
	# Test przypominania z zaszumionym wzorcem
	test_pattern = patterns[0]  # Wzorzec testowy
	noisy_pattern = add_noise(test_pattern, noise_level=0.3)
	
	print(f"\nWzorzec oryginalny: {test_pattern}")
	print(f"Wzorzec zaszumiony:  {noisy_pattern}")
	print(f"Podobieństwo: {pattern_overlap(test_pattern, noisy_pattern):.2f}")
	
	# Przypominanie wzorca
	recalled, energy_hist, iterations = network.recall_asynchronous(noisy_pattern)
	
	print(f"\nWzorzec odzyskany:   {recalled}")
	print(f"Liczba iteracji: {iterations}")
	print(f"Podobieństwo do oryginału: {pattern_overlap(test_pattern, recalled):.2f}")
	print(f"Energia końcowa: {energy_hist[-1]:.4f}")
	
	# Wykres energii
	plt.figure(figsize=(10, 4))
	plt.plot(energy_hist, 'b-o', markersize=4)
	plt.title('Zmiana energii podczas przypominania')
	plt.xlabel('Iteracja')
	plt.ylabel('Energia')
	plt.grid(True, alpha=0.3)
	plt.show()