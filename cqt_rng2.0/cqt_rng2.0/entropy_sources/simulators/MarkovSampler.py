from abc import ABC, abstractmethod
import numpy as np
from ...utils.utils import generate_markov1_bitlist



class MarkovBitSource(EntropySource):
    """Generates random bits using a 1-bit Markov chain model."""
    
    def __init__(self, length: int, P1: float = 0.5, phi1: float = 0.0):
        """
        Initializes the MarkovBitSource.

        Parameters:
            length (int): Length of the bit sequence to be generated.
            P1 (float): Probability of state to be 1.
            phi1 (float): Autocorrelation at lag 1.
        """
        self.length = length
        self.P1 = P1
        self.phi1 = phi1

    def sample(self, length: int = None) -> np.ndarray:
        """Generates a random bit sequence using the Markov model."""
        if length is None:
            length = self.length
        return np.array(generate_markov1_bitlist(length, self.P1, self.phi1))