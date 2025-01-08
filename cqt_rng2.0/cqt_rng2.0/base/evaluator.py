import numpy as np
from .rng import RNG
from .entropy_source import EntropySource
from .post_processor import PostProcessor
from collections import Counter




class Evaluator:


    
    """
    This is a class from which you can evaluate the randomness of your RNG System
    Using ExE, Autocorrelation and Entropy


    Parameters:
        entropy_source (EntropySource): The entropy source to sample from.
        postprocessor (PostProcessor): The post-processor.
        metrics: From the defined possible metrics
        lag: The lag that willbe uses if evaluating the autocorrelation, by default is 1
        entropy_n: The size of the pattern that the entropy will be counted on

    Examples:
        Evaluating the randomness of an RNG scheme that uses the BosonSampling as
        entropy source and the Von Neumann postprocessor::

            eval = Evaluator(BosonSampling(), VonNeumannPP(), ["ExE"])
            rng1.evaluate()
    """

    def __init__(
        self,
        entropy_source: EntropySource,
        postprocessor: PostProcessor,
        metrics=None,
        lag = 1,
        entropy_n = 1
    ):
        self.entropy_source = entropy_source
        self.postprocessor = postprocessor
        self.lag =lag
        self.entopy_n = entropy_n
        if metrics is None:
            self.metrics = ["ExE", "AutoCorr", "Entropy"]

    def _Exe(Input_Bitstring_len, Output_Bitstring_len):

        return Output_Bitstring_len/Input_Bitstring_len
    
    
    def _AutoCorr(self, bitlist):
        
        lag = self.lag
        bitlist = bitlist.tolist()
        n = len(bitlist)
        if lag >= n:
            raise ValueError("Lag must be less than the length of the bitlist.")

        bitlist = np.array(bitlist)

        # Calculate the mean of the bitlist
        mean = np.mean(bitlist)

        # Calculate the autocovariance and variance
        autocovariance = np.sum((bitlist[:n-lag] - mean) * (bitlist[lag:] - mean)) / (n - lag)
        variance = np.var(bitlist)

        # Calculate autocorrelation
        autocorrelation = autocovariance / variance

        return autocorrelation
    

    
    def Entropy(self, bitstring):

        n = self.entopy_n
        bitstring = bitstring.tolist()
        # Ensure bitstring length is divisible by n
        bitstring = bitstring[:len(bitstring) - (len(bitstring) % n)]

        # Group bits into n-bit chunks
        chunks = [tuple(bitstring[i:i + n]) for i in range(0, len(bitstring), n)]

        # Count occurrences of each chunk
        chunk_counts = Counter(chunks)

        # Calculate probabilities
        total_chunks = len(chunks)
        probabilities = np.array([count / total_chunks for count in chunk_counts.values()])

        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))

        # Normalize the entropy
        normalized_entropy = entropy / n

        return normalized_entropy

        
    def evaluate(self, length=1024):
        rng = RNG(self.entropy_source, self.postprocessor)
        bitstring = rng.generate(length)

        results = {}

        if "ExE" in self.metrics:
            input_len = rng.raw_length
            output_len = len(bitstring)
            results["ExE"] = self._Exe(input_len, output_len)

        if "AutoCorr" in self.metrics:
            results[f"Autocorrelation at lag: {self.lag}"] = self._AutoCorr(bitstring)

        if "Entropy" in self.metrics:
            results[f"Autocorrelation at size of: {self.lag}-bit"] = self.Entropy(bitstring)

        return results

        





