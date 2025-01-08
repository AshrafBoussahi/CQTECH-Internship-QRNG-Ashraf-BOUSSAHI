import numpy as np
from collections import deque
from ..base.post_processor import PostProcessor

class MKV1(PostProcessor):



    def __init__(self):
        pass

    
    def postprocess(self, bitstring, bits2= None):
        
        if len(bitstring) < 2:
            raise ValueError("Bitstring must have at least 2 bits for decorrelation.")
        
        if bits2 is not None:
            bitstring = bitstring.appens(bits2)

        # Initialize queues for previous bit history
        queue0, queue1 = deque(), deque()
        de_correlated_bits = [bitstring[0]]  # First bit is taken as is

        # Process bitstring starting from the second bit
        for i in range(1, len(bitstring)):
            current_bit = bitstring[i]
            previous_bit = bitstring[i - 1]

            # Route current bit to the appropriate queue
            if previous_bit == 0:
                queue0.append(current_bit)
            else:
                queue1.append(current_bit)

        # Merge queues into the output
        de_correlated_bits.extend(queue0)
        de_correlated_bits.extend(queue1)

        return np.array(de_correlated_bits, dtype=np.int8)