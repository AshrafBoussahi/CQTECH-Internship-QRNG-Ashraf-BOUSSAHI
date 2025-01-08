import numpy as np
from collections import deque
from ..base.post_processor import PostProcessor


class MKV2(PostProcessor):




    def __init__(self):
        pass


    
    def postprocess(self, bitstring, bits2= None):
        
        if len(bitstring) < 2:
            raise ValueError("Bitstring must have at least 2 bits for decorrelation.")
        
        if bits2 is not None:
            bitstring = bitstring.appens(bits2)

        # Initialize 4 queues for 2-bit history
        queues = {
            "00": deque(),
            "01": deque(),
            "10": deque(),
            "11": deque()
        }

        de_correlated_bits = [bitstring[0], bitstring[1]]  # First two bits are taken as is

        # Process bitstring starting from the third bit
        for i in range(2, len(bitstring)):
            current_bit = bitstring[i]
            previous_bits = f"{bitstring[i - 2]}{bitstring[i - 1]}"

            # Route current bit to the appropriate queue
            queues[previous_bits].append(current_bit)

        # Merge queues into the output
        for key in ["00", "01", "10", "11"]:
            de_correlated_bits.extend(queues[key])

        return np.array(de_correlated_bits, dtype=np.int8)
