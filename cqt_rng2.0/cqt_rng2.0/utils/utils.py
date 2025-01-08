from itertools import groupby, permutations
import numpy as np

def get_subm_idx(inp_state_array):
    indices = []
    for i in range(len(inp_state_array)):
        indices.extend([i] * inp_state_array[i])
    return indices

def decompose(n):
    if n == 1:
        return [[1]]
    elif n == 2:
        return [[1, 1], [2]]
    elif n == 3:
        return [[3], [2, 1], [1, 1, 1]]
    elif n == 4:
        return [[4], [3, 1], [2, 2], [2, 1, 1], [1, 1, 1, 1]]
    elif n == 5:
        return [[5], [4, 1], [3, 1, 1], [3, 2], [2, 1, 1, 1], [2, 2, 1], [1, 1, 1, 1, 1]]
    else:
        raise NotImplemented("Works only for n < 6 :)!")

def generate_output_states(total_photons, dim):
    output_states = []
    for i in decompose(total_photons):
        diff = [0] * (dim - len(i))
        i.extend(diff)
        l = list(permutations(i))
        l.sort()
        cleaned = list(l for l,_ in groupby(l))
        for c in cleaned:
            s = [str(i) for i in c]
            output_states.append("".join(s))
    return output_states



def generate_markov1_bitlist(length, P1=0.5, phi1=0.0):
    """
    Generates a bitlist using a 1-bit Markov chain model (MKV1).

    Args:
        length (int): Length of the generated bitlist.
        P1 (float): Probability of state to be 1.
        phi1 (float): Autocorrelation at lag 1.

    Returns:
        list: Generated bitlist.
    """
    # Calculate transition probabilities using the MKV1 model
    T01 = P1 * (1 - phi1)
    T11 = phi1 + T01
    T00 = 1 - T01
    T10 = 1 - T11

    # Validate probabilities
    if not (0 <= T01 <= 1 and 0 <= T11 <= 1):
        raise ValueError("Transition probabilities must be between 0 and 1.")

    # Initial state probabilities
    P0 = 1 - P1

    # Generate the bitlist
    bitlist = [np.random.choice([0, 1], p=[P0, P1])]

    for _ in range(1, length):
        current_state = bitlist[-1]
        if current_state == 0:
            next_bit = np.random.choice([0, 1], p=[T00, T01])
        else:
            next_bit = np.random.choice([0, 1], p=[T10, T11])
        bitlist.append(next_bit)

    return bitlist