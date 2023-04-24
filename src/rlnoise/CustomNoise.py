import numpy as np
from qibo.noise import DepolarizingError, NoiseModel, ThermalRelaxationError
from qibo import gates
from qibo.gates import ThermalRelaxationChannel,DepolarizingChannel
from rlnoise.dataset import CircuitRepresentation
from qibo.models import Circuit
class CustomNoiseModel(object):
    def __init__(self,time=0.07,lam=0.15, coherent_err=True):
        self.primitive_gates= ['RZ', 'RX','CZ']
        self.channels=['DepolarizingChannel','ThermalRelaxationChannel']
        self.qibo_noise_model=NoiseModel()
        self.Therm_on_gate=gates.RZ
        self.Depol_on_gate=gates.CZ
        self.time=time
        self.lam=lam
        self.qibo_noise_model.add(ThermalRelaxationError(t1=1,t2=1,time=self.time), self.Therm_on_gate)
        self.qibo_noise_model.add(DepolarizingError(self.lam),self.Depol_on_gate )
        self.coherent_err=coherent_err
        self.rep=CircuitRepresentation(primitive_gates=self.primitive_gates,noise_channels=self.channels,shape='3d')
    
    
    def apply(self,circuit):
        no_noise_circ_rep=self.rep.circuit_to_array(circuit)#SHAPE: (n_moments,n_qubits,encoding_dim=8)

        if self.coherent_err is False:
            
            noisy_circuit=self.qibo_noise_model.apply(circuit)
            
        else:
            nqubits=no_noise_circ_rep.shape[1]
            noisy_circuit=Circuit(nqubits, density_matrix=True)
            print('no noise BEFORE\n',no_noise_circ_rep)
            num_prim_gates=len(self.primitive_gates)
            for moment in range(no_noise_circ_rep.shape[0]):
                count=-1
                for qubit in range(no_noise_circ_rep.shape[1]):
                    gate_idx=no_noise_circ_rep[moment,qubit,:num_prim_gates].nonzero()[0]
                    
                    if len(gate_idx) != 0:
                        theta=no_noise_circ_rep[moment,qubit,num_prim_gates]
                        #print('gate idx: ',gate_idx)
                        #print('gate of type: ',self.rep.index2gate[int(gate_idx)])
                        if self.rep.index2gate[int(gate_idx)] is gates.RZ:
                            
                            print('no noise BEFORE\n',no_noise_circ_rep)
                            no_noise_circ_rep[moment,qubit,-2]=theta/2 #to be generalized (the epsilonZ is the -2 column)
                            noisy_circuit.add(gates.RZ(q=qubit,theta=theta))
                            noisy_circuit.add(gates.RZ(q=qubit,theta=theta/2))
                            if self.Depol_on_gate is gates.RZ:
                                noisy_circuit.add(DepolarizingChannel(lam=self.lam,q=qubit))
                                print('noisy circuit :',noisy_circuit.draw())
                            elif self.Therm_on_gate is gates.RZ:
                                noisy_circuit.add(ThermalRelaxationChannel(t1=1,t2=1,time=self.time,q=qubit))
                                print('noisy circuit :',noisy_circuit.draw())


                            print('no noise After\n',no_noise_circ_rep)
                        elif self.rep.index2gate[int(gate_idx)] is gates.RX:
                            no_noise_circ_rep[moment,qubit,-1]=no_noise_circ_rep[moment,qubit,num_prim_gates]/2
                            noisy_circuit.add(gates.RX(q=qubit,theta=theta))
                            noisy_circuit.add(gates.RX(q=qubit,theta=theta/2))                            
                            if self.Depol_on_gate is gates.RX:
                                noisy_circuit.add(DepolarizingChannel(lam=self.lam,q=qubit))
                                print('noisy circuit :',noisy_circuit.draw())
                            elif self.Therm_on_gate is gates.RX:
                                noisy_circuit.add(ThermalRelaxationChannel(t1=1,t2=1,time=self.time,q=qubit))
                                print('noisy circuit :',noisy_circuit.draw())


                        elif self.rep.index2gate[int(gate_idx)] is gates.CZ:
                            if count==-1:
                                count=qubit
                            else:
                                noisy_circuit.add(gates.CZ(q0=count,q1=qubit))
                                if self.Depol_on_gate is gates.CZ:
                                    noisy_circuit.add(DepolarizingChannel(lam=self.lam,q=(qubit,count)))
                                    print('noisy circuit :',noisy_circuit.draw())
                                elif self.Therm_on_gate is gates.CZ:
                                    noisy_circuit.add(ThermalRelaxationChannel(t1=1,t2=1,time=self.time,q=qubit))
                                    noisy_circuit.add(ThermalRelaxationChannel(t1=1,t2=1,time=self.time,q=count))
                                    print('noisy circuit :',noisy_circuit.draw())

            print('FINAL NOISY CIRCUIT\n')
            print(noisy_circuit.draw())
            print('Starting circuit: \n')
            print(circuit.draw())

            #circuit=self.rep.rep_to_circuit(no_noise_circ_rep)
            #print(circuit.draw())





            #print(circuit.draw())
            #print(no_noise_circ_rep)



            
        return noisy_circuit
        
