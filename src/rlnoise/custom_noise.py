import json
from pathlib import Path
from typing import List
from dataclasses import dataclass, field
from qibo.noise import DepolarizingError, NoiseModel, ThermalRelaxationError,ResetError
from qibo import gates
from qibo.models import Circuit

config_path=str(Path().parent.absolute())+'/src/rlnoise/config.json'
with open(config_path) as f:
    config = json.load(f)

noise_params = config['noise']

def string_to_gate(gate_string):   
    gate_str_low=gate_string.lower()
    if gate_str_low == 'none':
        return None
    elif gate_str_low == 'rx':
        gate=gates.RX
    elif gate_str_low == 'x':
        gate=gates.X
    elif gate_str_low == 'rz':
        gate=gates.RZ
    elif gate_str_low == 'cz':
        gate=gates.CZ
    elif gate_str_low == 'cnot':
        gate = gates.CNOT
    else:
    
        raise ValueError(f'Unrecognised gate {gate_str_low} in string_to_gate()')
    return gate

@dataclass
class CustomNoiseModel(object):
    primitive_gates: List = field(default_factory=lambda: noise_params['primitive_gates'])
    lam: float = noise_params['dep_lambda']
    p0: float = noise_params['p0']
    epsilon_x: float = noise_params['epsilon_x']
    epsilon_z: float = noise_params['epsilon_z']
    x_coherent_on_gate: List = field(default_factory=lambda: noise_params['x_coherent_on_gate'])
    z_coherent_on_gate: List = field(default_factory=lambda: noise_params['z_coherent_on_gate'])
    damping_on_gate: List = field(default_factory=lambda: noise_params['damping_on_gate'])
    depol_on_gate: List = field(default_factory=lambda: noise_params['depol_on_gate'])
        
       
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
        if apply_noise:
            simple_noisy_circuit=simple_noise.apply(circuit)
        else:
            simple_noisy_circuit=circuit
        noisy_circuit=Circuit(nqubits, density_matrix=True)
        for gate in simple_noisy_circuit.queue:
            noisy_circuit.add(gate)
            if self.x_coherent_on_gate is not None:
                for x_coherent_on_gate in self.x_coherent_on_gate:
                    if type(gate) == string_to_gate(x_coherent_on_gate):   
                        qubit=gate.qubits[0]                    
                        theta=self.epsilon_x*gate.init_kwargs['theta']
                        #print('Real theta %f, epsilon_x %f, theta coherent %f'%(gate.init_kwargs['theta'] ,self.noise_params["x"],theta))
                        noisy_circuit.add(gates.RX(q=qubit, theta=theta))
            if self.z_coherent_on_gate is not None:
                for z_coherent_on_gate in self.z_coherent_on_gate:
                    if type(gate) == string_to_gate(z_coherent_on_gate):
                        qubit=gate.qubits[0]
                        theta=self.epsilon_z*gate.init_kwargs['theta']
                        #print('Real theta %f, epsilon_z %f, theta coherent %f'%(gate.init_kwargs['theta'] ,self.noise_params["z"],theta))
                        noisy_circuit.add(gates.RZ(q=qubit, theta=theta))
        return noisy_circuit

    def check_gates_compatibility(self):
        primitive_gate_list=[]
        for gate_str in self.primitive_gates:
            primitive_gate_list.append(string_to_gate(gate_str))
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
                raise('Error: Attaching Damping channel to gate '+ damp_gate_str +' that is not one of the primitive gate')  
        for depol_gate_str in self.depol_on_gate:
            depol_gate=string_to_gate(depol_gate_str)
            if (depol_gate not in primitive_gate_list):
                raise('Error: Attaching Depolarizing channel to gate '+ depol_gate_str +' that is not one of the primitive gate')          
         
