from ...base.entropy_source import EntropySource
from ...utils.generate_haar_unitary import generate_haar_unitary
import pennylane as qml
import numpy as np
from ...utils.utils import generate_output_states



class ShiPL_SFSampler(EntropySource):
    """Simulates the Shi. and al. experiment on Strawberry Fields.

    Parameters:
        nb_modes (int, optional): Number of modes of the interferometer (should be an even number). Default to `6`.
        unitary_top (np.ndarray, optional): The unitary to apply on the top-half of the modes. Default to randomly generated haar matrix.
        unitary_bottom (np.ndarray, optional): The unitary to apply on the bottom-half of the modes. Default to randomly generated haar matrix.
        cut_off_dim = the maximum dimension of the fock space. Default to 4
    """

    def __init__(self, **kwargs):
        self.name = "ShiSFSampler"
        self.nb_modes = kwargs.get("nb_modes")
        self.unitary_top = kwargs.get("unitary_top")
        self.unitary_bottom = kwargs.get("unitary_bottom")
        self.cutoff_dim = kwargs.get("cut_off_dim")

        if self.cutoff_dim is None:
            self.cutoff_dim = 4

        if self.nb_modes is None:
            self.nb_modes = 6

        if self.nb_modes % 2 or self.nb_modes < 4:
            raise ValueError(
                "Wrong number of modes (nb_modes) expected to be even and higher than 3!"
            )

        if self.unitary_top is None:
            self.unitary_top = generate_haar_unitary(self.nb_modes // 2)

        if self.unitary_bottom is None:
            self.unitary_bottom = generate_haar_unitary(self.nb_modes // 2)

        if (
            np.shape(self.unitary_top)[0] != self.nb_modes // 2
            or np.shape(self.unitary_bottom)[0] != self.nb_modes // 2
            or np.shape(self.unitary_top)[1] != self.nb_modes // 2
            or np.shape(self.unitary_bottom)[1] != self.nb_modes // 2
        ):
            raise ValueError(
                f"Wrong unitary dimensions expect to be ({int(self.nb_modes / 2)},{int(self.nb_modes / 2)}) for top and bottom!"
            )

        self.dep_seq_len = self.nb_modes
        self.seq_len = self.nb_modes
        self.dev = qml.device("strawberryfields.fock", wires=self.nb_modes, cutoff_dim=self.cutoff_dim)

    def _successful_entanglement(self, sample):
        nb_modes = np.size(sample)
        return (
            np.sum(sample[: nb_modes // 2]) == 1
            and np.sum(sample[nb_modes // 2 :]) == 1
        )

    def _plsf_simulator(self):

        @qml.qnode(self.dev)
        def circuit():
            mid_l = self.nb_modes // 2 - 1
            mid_h = self.nb_modes // 2

            # Two-photon entanglement source
            qml.FockState(1, wires=mid_l)
            qml.FockState(1, wires=mid_h)

            # Beam splitters for entanglement
            qml.Beamsplitter(phi=np.pi / 2, theta=np.pi / 4, wires=[mid_l - 1, mid_l])
            qml.Beamsplitter(phi=np.pi / 2, theta=np.pi / 4, wires=[mid_h, mid_h + 1])

            # SWAP operation
            qml.Beamsplitter(phi=np.pi / 2, theta=np.pi / 2, wires=[mid_l, mid_h])

            # Interferometer
            qml.InterferometerUnitary(self.unitary_top, wires=list(range(mid_l + 1)))
            qml.InterferometerUnitary(self.unitary_bottom, wires=list(range(mid_h, self.nb_modes)))

            # Measurement
            return qml.probs(wires=range(self.nb_modes))
        
        probs = circuit().reshape([self.cutoff_dim] * self.nb_modes)

        measure_states = [tuple(map(int, list(item))) for item in generate_output_states(2, self.nb_modes)]

        probabilities = [probs[i].item() for i in measure_states]

        probabilities /= np.sum(probabilities)

        return np.asarray([1 if x > 1 else x for x in list(measure_states[np.random.choice(range(len(measure_states)), p=probabilities)])])
        

    def _run_experiment(self):
        ran_once = False
        while not ran_once or self._successful_entanglement(sample):
            sample = self._plsf_simulator()
            ran_once = True
        return sample

    def sample(self, length):
        shots = length // self.seq_len
        ret = np.array([])
        for _ in range(shots):
            ret = np.append(ret, self._run_experiment())

        return np.copy(ret[:length]).astype(np.int8)