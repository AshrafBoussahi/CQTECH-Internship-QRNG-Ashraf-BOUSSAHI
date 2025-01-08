import pennylane as qml
import numpy as np

class UniversalQCSampler:
    """
    Sample taken from PennyLane's default.qubit simulator (using the Qubit-based approach)
    """

    def __init__(self, **kwargs):
        
        self.name = "UniversalQCSampler"
        self.nb_qubits = kwargs.get("nb_qubits", 5)
        self.operation = kwargs.get("operation", "hadamard")
        self.angle = kwargs.get("angle", None)

        if self.operation == "rotation" and self.angle is None:
            raise ValueError("You must define an angle when choosing the 'rotation' operation.")

        # Set up the device with a default number of shots
        self.default_shots = 1000
        self.dev = qml.device("default.qubit", wires=self.nb_qubits, shots=self.default_shots)

    def simulate(self, length):
        shots = np.max([length // self.nb_qubits, 2])
    
        @qml.qnode(self.dev)
        def circuit():
            for i in range(self.nb_qubits):
                if self.operation == "hadamard":
                    qml.Hadamard(wires=i)
                elif self.operation == "rotation":
                    qml.RY(self.angle, wires=i)
                else:
                    raise NotImplementedError("Defined operation not implemented yet!")
            #print(qml.draw(circuit)())
            return [qml.sample(qml.PauliZ(wires=i)) for i in range(self.nb_qubits)]
        
        #(qml.draw(circuit)())
        # Run the circuit multiple times to get the desired number of samples
        result = []
        for _ in range(shots):
            result.extend(circuit())
    
        return np.array(result)


    def sample(self, length):
        memory = self.simulate(length).flatten()
        # Map -1 to 0 and +1 to 1
        memory = (memory + 1) // 2
        return memory[:length]

