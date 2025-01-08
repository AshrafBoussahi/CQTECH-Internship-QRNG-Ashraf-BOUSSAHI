from ..base.post_processor import PostProcessor
import numpy as np


class NoPostProcess(PostProcessor):
    def __init__(self):
        pass

    def postprocess(self, sample_1, sample_2=None):

        if sample_2 is None:
            output = np.copy(np.append(sample_1)).astype(np.int8)
        else:
            output = np.copy(np.append(sample_1, sample_2)).astype(np.int8)

        return output
