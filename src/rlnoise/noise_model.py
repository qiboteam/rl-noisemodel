import json
from typing import List
from dataclasses import dataclass
from qibo.noise import DepolarizingError, NoiseModel, ResetError
from qibo import gates
from qibo.models import Circuit

def string_to_gate(gate_string):   
    gate_str_low=gate_string.lower()
    if gate_str_low == 'none':
        return None
    elif gate_str_low == 'rx':
        gate=gates.RX
    elif gate_str_low == 'rz':
        gate=gates.RZ
    elif gate_str_low == 'cz':
        gate=gates.CZ
    else:
        raise ValueError(f'Unrecognised gate {gate_str_low} in string_to_gate()')
    return gate

@dataclass
class CustomNoiseModel(object):
    """
    Define a custom noise model for the generation of the datasets.
    """
    config_file: str
    primitive_gates: List = None
    lam: float = None
    p0: float = None
    epsilon_x: float = None
    epsilon_z: float = None
    x_coherent_on_gate: List = None
    z_coherent_on_gate: List = None
    damping_on_gate: List = None
    depol_on_gate: List = None

    def __post_init__(self):
        with open(self.config_file) as f:
            self.noise_params = json.load(f)['noise']
        self.primitive_gates = self.noise_params['primitive_gates']
        self.lam = self.noise_params['dep_lambda']
        self.p0 = self.noise_params['p0']
        self.epsilon_x = self.noise_params['epsilon_x']
        self.epsilon_z = self.noise_params['epsilon_z']
        self.x_coherent_on_gate = self.noise_params['x_coherent_on_gate']
        self.z_coherent_on_gate = self.noise_params['z_coherent_on_gate']
        self.damping_on_gate = self.noise_params['damping_on_gate']
        self.depol_on_gate = self.noise_params['depol_on_gate']
        
       
    def apply(self, circuit):
        self.check_gates_compatibility()
        nqubits = circuit.nqubits
        apply_noise = False
        simple_noise = NoiseModel()
        for damping_gate in self.damping_on_gate :
            target_gate = string_to_gate(damping_gate)
            if target_gate is not None:
                simple_noise.add(ResetError(p0=self.p0, p1=0), target_gate)
                apply_noise = True
        for depol_gate in self.depol_on_gate:
            target_gate=string_to_gate(depol_gate)
            if target_gate is not None:
                simple_noise.add(DepolarizingError(self.lam),target_gate )
                apply_noise=True
                
        simple_noisy_circuit = simple_noise.apply(circuit) if apply_noise else circuit
        noisy_circuit=Circuit(nqubits, density_matrix=True)
        for gate in simple_noisy_circuit.queue:
            noisy_circuit.add(gate)
            if self.x_coherent_on_gate is not None:
                for x_coherent_on_gate in self.x_coherent_on_gate:
                    if type(gate) == string_to_gate(x_coherent_on_gate):   
                        qubit=gate.qubits[0]                    
                        theta=self.epsilon_x*gate.init_kwargs['theta']
                        noisy_circuit.add(gates.RX(q=qubit, theta=theta))
            if self.z_coherent_on_gate is not None:
                for z_coherent_on_gate in self.z_coherent_on_gate:
                    if type(gate) == string_to_gate(z_coherent_on_gate):
                        qubit=gate.qubits[0]
                        theta=self.epsilon_z*gate.init_kwargs['theta']
                        noisy_circuit.add(gates.RZ(q=qubit, theta=theta))
        return noisy_circuit

    def check_gates_compatibility(self):
        primitive_gate_list = [
            string_to_gate(gate_str) for gate_str in self.primitive_gates
        ]
        primitive_gate_list.append(None)
        for epsilon_z_gate in self.z_coherent_on_gate:
            gate=string_to_gate(epsilon_z_gate)
            if (gate not in primitive_gate_list):
                raise('Error: Attaching epsilon_z channel to gate not present in PrimitiveGates')
        for epsilon_x_gate in self.z_coherent_on_gate:
            gate=string_to_gate(epsilon_x_gate)
            if (gate not in primitive_gate_list):
                raise('Error: Attaching epsilon_x channel to gate not present in PrimitiveGates')
        for damp_gate_str in self.damping_on_gate:
            damp_gate=string_to_gate(damp_gate_str)
            if (damp_gate not in primitive_gate_list):
                raise f'Error: Attaching Damping channel to gate {damp_gate_str} that is not one of the primitive gate'
        for depol_gate_str in self.depol_on_gate:
            depol_gate=string_to_gate(depol_gate_str)
            if (depol_gate not in primitive_gate_list):
                raise f'Error: Attaching Depolarizing channel to gate {depol_gate_str} that is not one of the primitive gate'          
         
