from ..base.post_processor import PostProcessor
from .von_neumann_pp import VonNeumannPP
from .cqt_pp import CQTPP
import numpy as np

class IterCQTPP(PostProcessor):
    """Implementation of IterCQTPP
    
    Parameters:
        dep_seq_len (int): The length of the dependent sequences.
        it (int): The number of iterations for the post-processing.
        
    """
    
    def __init__(self, **kwargs):
        self.__dep_seq_len = kwargs.get("dep_seq_len")
        self.__iterations = kwargs.get("it")
        

        if self.__dep_seq_len is None:
            self.__dep_seq_len = 1
        if self.__iterations is None:
            self.__iterations = 1
        

    def postprocess(self, sample_1, sample_2 = None) -> np.ndarray:
        
        ouput = np.array([], dtype=np.int8)
        if sample_2 is None:
            sample_1, sample_2 = sample_1[:(len(sample_1)//2)], sample_1[(len(sample_1)//2):]

        oui, dii = CQTPP(dep_seq_len=self.__dpq).postprocess2(sample_1)
        out_final = oui
        for _ in range(self.__iterations - 1):
            oui, dii = CQTPP(dep_seq_len=self.__dpq).postprocess2(dii)
            out_final = np.append(out_final, oui)

        return out_final
