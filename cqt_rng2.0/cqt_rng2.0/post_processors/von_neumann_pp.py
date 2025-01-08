from ..base.post_processor import PostProcessor
import numpy as np


class VonNeumannPP(PostProcessor):
    def __init__(self):
        pass

    def postprocess(self, sample_1, sample_2 = None):

        if sample_2 is None:
            sample_1, sample_2 = sample_1[:(len(sample_1)//2)], sample_1[(len(sample_1)//2):]

        min_length = min(len(sample_1), len(sample_2))
        sample_1 = sample_1[:min_length]
        sample_2 = sample_2[:min_length]

        bits_1 = np.ravel(np.array(sample_1) == 0).astype(np.int8)
        bits_2 = np.ravel(np.array(sample_2) == 0).astype(np.int8)

        arr = np.where(bits_1 > bits_2, np.zeros_like(bits_1), np.ones_like(bits_1))
        arr = np.where(bits_1 == bits_2, np.nan * np.ones_like(bits_1), arr)

        return arr[~np.isnan(arr)].astype(np.int8)
