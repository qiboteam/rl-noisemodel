import numpy as np
from qibo.noise import DepolarizingError, NoiseModel, ThermalRelaxationError
from qibo import gates
from rlnoise.dataset import CircuitRepresentation

class CustomNoiseModel(object):
    def __init__(self,time=0.07,lam=0.15, coherent_err=True):
        self.primitive_gates= ['RZ', 'RX','CZ']
        self.channels=['DepolarizingChannel','ThermalRelaxationChannel']
        self.qibo_noise_model=NoiseModel()
        self.Therm_on_gate=gates.RZ
        self.Depol_on_gate=gates.CZ
        self.qibo_noise_model.add(ThermalRelaxationError(t1=1,t2=1,time=time), self.Therm_on_gate)
        self.qibo_noise_model.add(DepolarizingError(lam),self.Depol_on_gate )
        self.coherent_err=coherent_err
        self.rep=CircuitRepresentation(primitive_gates=self.primitive_gates,noise_channels=self.channels,shape='3d')
    
    
    def apply(self,circuit):
        no_noise_circ_rep=self.rep.circuit_to_array(circuit)#SHAPE: (n_moments,n_qubits,encoding_dim=8)

        if self.coherent_err is False:
            noisy_circuit=self.qibo_noise_model.apply(circuit)
            
        else:
            
            for moment in range(no_noise_circ_rep.shape[0]):
                for qubit in range(no_noise_circ_rep.shape[1]):
                    gate_idx=no_noise_circ_rep[moment,qubit,:len(self.primitive_gates)].nonzero()[0]
                    if len(gate_idx) is not 0:
                        print('gate idx: ',gate_idx)
                        print('gate of type: ',self.rep.index2gate[int(gate_idx)])
                        if self.rep.index2gate[int(gate_idx)] is gates.RZ:
                            print('no noise BEFORE\n',no_noise_circ_rep)
                            no_noise_circ_rep[moment,qubit,-2]=1
                            print('no noise After\n',no_noise_circ_rep)





            #print(circuit.draw())
            #print(no_noise_circ_rep)



            
        return noisy_circuit
        
