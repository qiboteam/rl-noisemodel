import json
import numpy as np
from configparser import ConfigParser
from qibo.noise import DepolarizingError, NoiseModel, ThermalRelaxationError,ResetError
from qibo import gates
from qibo.models import Circuit

def string_to_gate(gate_string):   
    if gate_string == 'rx' or gate_string== 'RX':
        gate=gates.RX
    elif gate_string == 'rz' or gate_string== 'RZ':
        gate=gates.RZ
    elif gate_string == 'cz' or gate_string== 'CZ':
        gate=gates.CZ
    else:
        print('Error: unrecognised gate in string_to_gate()')
        exit()
    return gate

params=ConfigParser()

params.read("src/rlnoise/config.ini") 
primitive_gates= json.loads(params.get('noise','primitive_gates'))
channels= json.loads(params.get('noise','channels'))
time=params.getfloat('noise','thermal_time')
t1=params.getfloat('noise','t1')
t2=params.getfloat('noise','t2')
lam=params.getfloat('noise','dep_lambda')
p0=params.getfloat('noise','p0')
coherent_err=params.getboolean('noise','coherent_noise')
std_noise=params.getboolean('noise','std_noise')   
x_coherent_on_gate=string_to_gate(params.get('noise','x_coherent_on_gate'))        
z_coherent_on_gate=string_to_gate(params.get('noise','z_coherent_on_gate'))
epsilonX=params.getfloat('noise','epsilon_x')
epsilonZ=params.getfloat('noise','epsilon_z')
Damping_on_gate=json.loads(params.get('noise','damping_on_gate'))
Depol_on_gate=json.loads(params.get('noise','depol_on_gate'))

class CustomNoiseModel(object):

    def __init__(self,primitive_gates=primitive_gates,channels=channels,t1=t1,t2=t2,
                 lam=lam,p0=p0,coherent_err=coherent_err,std_noise=std_noise,
                 x_coherent_on_gate=x_coherent_on_gate,z_coherent_on_gate=z_coherent_on_gate,
                 epsilonX=epsilonX,epsilonZ=epsilonZ,Damping_on_gate=Damping_on_gate,Depol_on_gate=Depol_on_gate):
        self.primitive_gates= primitive_gates
        self.channels= channels
        self.time=time
        self.t1=t1
        self.t2=t2
        self.lam=lam
        self.p0=p0
        self.coherent_err=coherent_err
        self.std_noise=std_noise
        self.x_coherent_on_gate=x_coherent_on_gate   
        self.z_coherent_on_gate=z_coherent_on_gate
        self.epsilonX=epsilonX
        self.epsilonZ=epsilonZ
        self.qibo_noise_model=NoiseModel()
        
        if self.std_noise is True: 
            #self.Therm_on_gate=gates.RZ
            self.Damping_on_gate=Damping_on_gate
            self.Depol_on_gate=Depol_on_gate
            self.check_gates_compatibility()
            for dampin_gate in self.Damping_on_gate:
                target_gate=string_to_gate(dampin_gate)
                self.qibo_noise_model.add(ResetError(p0=self.p0, p1=0), target_gate)
            for depol_gate in self.Depol_on_gate:
                target_gate=string_to_gate(depol_gate)
                self.qibo_noise_model.add(DepolarizingError(self.lam),target_gate )
       
    def apply(self, circuit):
        nqubits=circuit.nqubits     
        if self.std_noise is True:
            simple_noisy_circuit=self.qibo_noise_model.apply(circuit)
        else:
            simple_noisy_circuit=circuit
        if self.coherent_err:
            noisy_circuit=Circuit(nqubits, density_matrix=True)
            for gate in simple_noisy_circuit.queue:
                noisy_circuit.add(gate)
                if type(gate) is self.x_coherent_on_gate:   
                    qubit=gate.qubits[0]                    
                    theta=self.epsilonX*gate.init_kwargs['theta'] 
                    noisy_circuit.add(gates.RX(q=qubit, theta=theta))
                if type(gate) is self.z_coherent_on_gate:
                    qubit=gate.qubits[0]
                    theta=self.epsilonZ*gate.init_kwargs['theta'] 
                    noisy_circuit.add(gates.RZ(q=qubit, theta=theta))
        else:
            noisy_circuit=simple_noisy_circuit
        return noisy_circuit

    def check_gates_compatibility(self):
        primitive_gate_list=[]
        for gate_str in self.primitive_gates:
            primitive_gate_list.append(string_to_gate(gate_str))
        for damp_gate_str in self.Damping_on_gate:
            damp_gate=string_to_gate(damp_gate_str)
            if damp_gate not in primitive_gate_list:
                print('Error: Attaching Damping channel to gate '+damp_gate_str+' that is not one of the primitive gate')
                exit()
        for depol_gate_str in self.Depol_on_gate:
            depol_gate=string_to_gate(depol_gate_str)
            if depol_gate not in primitive_gate_list:
                print('Error: Attaching Depolarizing channel to gate '+depol_gate_str+' that is not one of the primitive gate')
                exit()                
         
