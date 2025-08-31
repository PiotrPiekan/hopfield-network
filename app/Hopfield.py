import numpy as np

class Hopfield():
	"""
	Implementacja sieci Hopfielda.

	Attributes
	----------
	weights : ndarray
		Dwuwymiarowa macierz wag połączeń neuronów.
	biases : ndarray
		Jednowymiarowa macierz wyrazów wolnych.
	size : int
		Rozmiar sieci (liczba neuronów).
	"""

	def __init__(self, size, weights=None, biases=None):
		"""
		Inicjalizuje sieć Hopfielda.

		Parameters
		----------
		size : int
			Liczba neuronów w sieci.
		weights : ndarray, optional
			Macierz wag. Jeśli None, inicjalizowana zerami.
		biases : ndarray, optional
			Wektor biasów. Jeśli None, inicjalizowany zerami.
		"""
		self.size = size
		
		if weights is not None and biases is not None:
			self.weights = weights
			self.biases = biases
		else:
			self.weights = np.zeros((size, size))
			self.biases = np.zeros(size)

	def train(self, patterns):
		"""
		Uczy sieć na podstawie listy wzorców.

		Parameters
		----------
		patterns : list of ndarray
			Lista wzorców do zapamiętania. Każdy wzorzec to tablica numpy
			z wartościami 1/-1 o tym samym rozmiarze.
		"""
		self.weights.fill(0)
		self.biases.fill(0)

		for i in range(len(patterns)):
			pattern = patterns[i].flatten()
			self.weights += np.outer(pattern, pattern)
			self.biases += pattern

		np.fill_diagonal(self.weights, 0)

		self.weights /= patterns[0].size
		self.biases /= patterns[0].size

	def energy(self, state):
		"""
		Oblicza energię stanu sieci.

		Parameters
		----------
		state : ndarray
			Stan sieci (wektor neuronów).

		Returns
		-------
		float
			Wartość energii dla danego stanu.
		"""
		weight_energy = np.dot(np.dot(self.weights, state).T, state)
		biases_energy = np.dot(self.biases, state)

		return -0.5 * weight_energy - biases_energy
	
	def recall(self, input_pattern, synchronous=True, max_iterations=100, energy_tol=0):
		"""
		Odtwarza wzorzec na podstawie wejściowego wzorca.

		Parameters
		----------
		input_pattern : ndarray
			Wzorzec wejściowy do odtworzenia.
		synchronous : bool, default True
			True - Aktualizacja wszystkich neuronów jednocześnie 
			False - Aktualizacja neuronóœ w losowej kolejności.
		max_iterations : int, default 100
			Maksymalna liczba iteracji.
		energy_tol : float, default 0
			Tolerancja zmiany energii do zatrzymania algorytmu.

		Returns
		-------
		tuple
			(states_history, energy_history) - historia stanów i energii.
		"""
		state = input_pattern.flatten()
		states_history = [state.copy()]
		energy_history = [self.energy(state)]

		for iteration in range(max_iterations):
			if synchronous:
				activation = np.dot(self.weights, state) + self.biases
				state = np.where(activation >= 0, 1, -1)
			else:
				idx = np.random.permutation(self.size)
				for i in idx:
					activation = np.dot(self.weights[i], state) + self.biases[i]
					state[i] = 1 if activation >= 0 else -1

			states_history.append(state.copy())
			energy_history.append(self.energy(state))

			if abs(energy_history[-1] - energy_history[-2]) < energy_tol:
				break

		return states_history, energy_history

if __name__ == "__main__":
	import app
	app.main()