import numpy as np
from qibo.noise import DepolarizingError, NoiseModel, ThermalRelaxationError
from qibo import gates
from qibo.gates import ThermalRelaxationChannel,DepolarizingChannel
from rlnoise.datasetv2 import CircuitRepresentation
from qibo.models import Circuit
class CustomNoiseModel(object):

    def __init__(self,time=0.07,lam=0.05, coherent_err=False,std_noise=True):
        self.primitive_gates= ['RZ', 'RX']#,'RY']
        self.channels=['DepolarizingChannel','ThermalRelaxationChannel']
        self.time=time
        self.t1=1
        self.t2=1
        self.lam=lam
        self.std_noise=std_noise
        self.qibo_noise_model=NoiseModel()

        if self.std_noise is True: 
            self.Therm_on_gate=gates.RZ
            self.Depol_on_gate=gates.RX
            self.qibo_noise_model.add(ThermalRelaxationError(t1=self.t1,t2=self.t2,time=self.time), self.Therm_on_gate)
            self.qibo_noise_model.add(DepolarizingError(self.lam),self.Depol_on_gate )
        else:
            self.Therm_on_gate=None
            self.Depol_on_gate=None

        self.coherent_err=coherent_err
        self.rep=CircuitRepresentation(primitive_gates=self.primitive_gates,noise_channels=self.channels,shape='3d')

        self.epsilonZ=0.15
        self.epsilonX=0.3
        
    def apply(self,circuit):
        no_noise_circ_rep=self.rep.circuit_to_array(circuit)#SHAPE: (n_moments,n_qubits,encoding_dim=8)

        if self.coherent_err is False:
            noisy_circuit=self.qibo_noise_model.apply(circuit)
            
        else:
            nqubits=no_noise_circ_rep.shape[1]
            noisy_circuit=Circuit(nqubits, density_matrix=True)
            num_prim_gates=len(self.primitive_gates)
            for moment in range(no_noise_circ_rep.shape[0]):
                count=-1
                for qubit in range(no_noise_circ_rep.shape[1]):
                    gate_idx=no_noise_circ_rep[moment,qubit,:num_prim_gates].nonzero()[0]
                    
                    if len(gate_idx) != 0:
                        theta=no_noise_circ_rep[moment,qubit,num_prim_gates]
                        if self.rep.index2gate[int(gate_idx)] is gates.RZ:                           
                            no_noise_circ_rep[moment,qubit,-2]=theta/2 #to be generalized (the epsilonZ is the -2 column)
                            noisy_circuit.add(gates.RZ(q=qubit,theta=theta))
                            if self.epsilonZ !=0:
                                noisy_circuit.add(gates.RZ(q=qubit,theta=theta*self.epsilonZ)) #epsilon is a parameter that describes how strong is the coherent noise
                            if self.Depol_on_gate is gates.RZ:
                                noisy_circuit.add(DepolarizingChannel(lam=self.lam,q=qubit))
                            elif self.Therm_on_gate is gates.RZ:
                                noisy_circuit.add(ThermalRelaxationChannel(t1=self.t1,t2=self.t2,time=self.time,q=qubit))
                        elif self.rep.index2gate[int(gate_idx)] is gates.RX:
                            no_noise_circ_rep[moment,qubit,-1]=no_noise_circ_rep[moment,qubit,num_prim_gates]/2 #to be generalized (the epsilonX is the -1 column)
                            noisy_circuit.add(gates.RX(q=qubit,theta=theta))
                            if self.epsilonX !=0:
                                noisy_circuit.add(gates.RX(q=qubit,theta=theta*self.epsilonX))     #epsilon is a parameter that describes how strong is the coherent noise                       
                            if self.Depol_on_gate is gates.RX:
                                noisy_circuit.add(DepolarizingChannel(lam=self.lam,q=qubit))
                            elif self.Therm_on_gate is gates.RX:
                                noisy_circuit.add(ThermalRelaxationChannel(t1=self.t1,t2=self.t2,time=self.time,q=qubit))
                        elif self.rep.index2gate[int(gate_idx)] is gates.CZ:
                            if count==-1:
                                count=qubit
                            else:
                                noisy_circuit.add(gates.CZ(q0=count,q1=qubit))
                                if self.Depol_on_gate is gates.CZ:
                                    noisy_circuit.add(DepolarizingChannel(lam=self.lam,q=(qubit,count)))
                                elif self.Therm_on_gate is gates.CZ:
                                    noisy_circuit.add(ThermalRelaxationChannel(t1=self.t1,t2=self.t2,time=self.time,q=qubit))
                                    noisy_circuit.add(ThermalRelaxationChannel(t1=self.t1,t2=self.t2,time=self.time,q=count))

            '''
            print('Applying coherent noise, Depolarizing (CZ) Thermal (RZ), on circuit: ')
            print(circuit.draw())
            print('FINAL NOISY CIRCUIT')
            print(noisy_circuit.draw(),'\n')     
            '''                     
        return noisy_circuit
        
