import numpy as np

from ..base.post_processor import PostProcessor

class Combined_PP(PostProcessor):



    def __init__(self, post_processor1 : PostProcessor, post_processor2 : PostProcessor):
        self.postprocessor1 = post_processor1
        self.postprocessor2 = post_processor2

    ''' This processor apply 2 sucessive post processors: post_processor1 and post_processor2 respectively'''

    
    def postprocess(self, bitstring, bits2= None):
        
        
        if bits2 is not None:
            bitstring = bitstring.appens(bits2)

        result1 = self.postprocessor1.postprocess(bitstring, bits2)

        return self.postprocessor2.postprocess(result1)