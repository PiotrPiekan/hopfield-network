import numpy as np
from typing import List

class Hopfield():
    """
        Implementacja sieci Hopfielda

        Attributes:
        ----------
        weights : ndarray[float, float]
            Dwuwymiarowa macierz wag połączeń neuronów
        biases : ndarray[float]
            Jednowymiarowa macierz wyrazów wolnych
        state : ndarray[int]
            Aktualny stan aktywacji neuronów (-1 / 1)
    """

    def __init__(self, size: int):
        """
            Inicjalizacja sieci Hopfielda

            Arguments
            ----------
            size : int
                Rozmiar sieci (liczba neuronów)
        """

        self.size = size
        self.weights = np.zeros((size, size))
        self.biases = np.zeros(size)

    def train(self, patterns: List[np.array]):
        """
            Uczenie sieci na podstawie listy wzorców.
            Wzorzec jest n-wymiarową tablicą numpy z wartościami 1 / -1
            Każdy wzorzec musi mieć tę samą liczbę wartości.

            Arguments
            ----------
            patterns : list[ndarray]
                Lista wzorców do zapamiętania.
        """

        for pattern in patterns:
            pattern = pattern.reshape(-1, 1)
            self.weights += np.outer(pattern, pattern)
            self.biases += pattern

        np.fill_diagonal(self.weights, 0)

        self.weights /= len(patterns)
        self.biases /= len(patterns)

    def energy(self):
        """
            Oblicza energię aktualnego stanu sieci

            Returns
            ----------
            float
                Energia
        """
        
        weight_energy = np.dot(np.dot(self.weights, self.state).T, self.state)
        biases_energy = np.dot(self.biases, self.state)

        return -0.5 * weight_energy - biases_energy
    
import matplotlib.pyplot as plt

print()
plt.subplot()