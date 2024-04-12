import numpy as np
from qibo import gates
from qibo.models.circuit import Circuit
from qibo.transpiler.unroller import Unroller, NativeGates
from qibo.transpiler.optimizer import Rearrange
from qibo.quantum_info import trace_distance as qibo_trace_distance, fidelity
from scipy.linalg import sqrtm

def trace_distance(rho1,rho2):
    """Compute the trace distance between two density matrices."""
    return qibo_trace_distance(rho1,rho2)

def qibo_compute_fidelity(density_matrix0, density_matrix1):
    """Compute the fidelity for two density matrices (pure or mixed states).

    .. math::
            F( \rho , \sigma ) = -\text{Tr}( \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}})^2
    """
    return fidelity(density_matrix0, density_matrix1)

def compute_fidelity(density_matrix0, density_matrix1):
    """Compute the fidelity for two density matrices (pure or mixed states).

    .. math::
            F( \rho , \sigma ) = -\text{Tr}( \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}})^2
    """
    sqrt_mat1_mat2 = sqrtm(density_matrix0 @ density_matrix1)
    trace = np.real(np.trace(sqrt_mat1_mat2)**2)
    if trace > 1:
        trace=1
    return trace

def test_avg_fidelity(rho1,rho2):
    fidelity = []
    for i in range(len(rho1)):
        print(i, "fidelity: ", compute_fidelity(rho1[i],rho2[i]))
        fidelity.append(compute_fidelity(rho1[i],rho2[i]))
    avg_fidelity = np.array(fidelity).mean()
    return avg_fidelity

def u3_dec(gate):
    """Decompose a U3 gate into RZ and RX gates."""
    # t, p, l = gate.parameters
    params = gate.parameters
    t = params[0]
    p = params[1]
    l = params[2]
    decomposition = []
    if l != 0.0:
        decomposition.append(gates.RZ(gate.qubits[0], l))
    decomposition.append(gates.RX(gate.qubits[0], np.pi/2, 0))
    if t != -np.pi:
        decomposition.append(gates.RZ(gate.qubits[0], t + np.pi))
    decomposition.append(gates.RX(gate.qubits[0], np.pi/2, 0))
    if p != -np.pi:
        decomposition.append(gates.RZ(gate.qubits[0], p + np.pi))
    return decomposition

def unroll_circuit(circuit):
    from qibo.transpiler.unitary_decompositions import u3_decomposition
    natives = NativeGates.U3 | NativeGates.CZ
    unroller = Unroller(native_gates = natives)
    optimizer = Rearrange()

    unrolled_circuit = unroller(circuit)
    opt_circuit = optimizer(unrolled_circuit)
    queue = opt_circuit.queue
    final_circuit = Circuit(circuit.nqubits, density_matrix=True)
    for gate in queue:
        if isinstance(gate, gates.CZ):
            final_circuit.add(gate)
        elif isinstance(gate, gates.RZ):
            final_circuit.add(gate)
        elif isinstance(gate, gates.Unitary):
            matrix = gate.matrix()
            u3_gate = gates.U3(gate.qubits[0], *u3_decomposition(matrix))
            decomposed = u3_dec(u3_gate)
            for decomposed_gate in decomposed:
                final_circuit.add(decomposed_gate)
        elif isinstance(gate, gates.U3):
            decomposed = u3_dec(gate)
            for decomposed_gate in decomposed:
                final_circuit.add(decomposed_gate)
    return final_circuit


def grover():
    """Creates a Grover circuit with 3 qubits.
    The circuit searches for the 11 state, the last qubit is ancillary"""
    circuit = Circuit(3, density_matrix=True)
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RX(0, np.pi/2))
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RZ(1, np.pi/2))
    circuit.add(gates.RX(1, np.pi/2))
    circuit.add(gates.RZ(1, np.pi/2))
    circuit.add(gates.RX(2, np.pi))
    circuit.add(gates.RZ(2, np.pi/2))
    circuit.add(gates.RX(2, np.pi/2))
    circuit.add(gates.RZ(2, np.pi/2))
    #Toffoli
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.RX(2, -np.pi / 4))
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.RX(2, np.pi / 4))
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.RX(2, -np.pi / 4))
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.RX(2, np.pi / 4))
    circuit.add(gates.RZ(1, np.pi / 4))
    circuit.add(gates.RZ(1, np.pi/2))
    circuit.add(gates.RX(1, np.pi/2))
    circuit.add(gates.RZ(1, np.pi/2))
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.RZ(0, np.pi / 4))
    circuit.add(gates.RX(1, -np.pi / 4))
    circuit.add(gates.CZ(0, 1))
    ###
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RX(0, np.pi/2))
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RX(0, np.pi))
    circuit.add(gates.RX(1, np.pi))
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.RX(0, np.pi))
    circuit.add(gates.RX(1, np.pi))
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RX(0, np.pi/2))
    circuit.add(gates.RZ(0, np.pi/2))
    circuit.add(gates.RZ(1, np.pi/2))
    circuit.add(gates.RX(1, np.pi/2))
    circuit.add(gates.RZ(1, np.pi/2))
    return circuit

def qft():
    circuit = Circuit(3, density_matrix=True)
    circuit.add(gates.H(0))
    #Gate S
    circuit.add(gates.U3(1, 0, 0, np.pi / 4))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.U3(1, 0, 0, -np.pi / 4))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.RZ(0, np.pi / 4))
    #Gate T
    circuit.add(gates.U3(2, 0, 0, np.pi / 8))
    circuit.add(gates.CNOT(0, 2))
    circuit.add(gates.U3(2, 0, 0, -np.pi / 8))
    circuit.add(gates.CNOT(0, 2))
    circuit.add(gates.RZ(0, np.pi / 8))

    circuit.add(gates.H(1))
    #Gate S
    circuit.add(gates.U3(2, 0, 0, np.pi / 4))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.U3(2, 0, 0, -np.pi / 4))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.RZ(1, np.pi / 4))

    circuit.add(gates.H(2))
    
    return circuit