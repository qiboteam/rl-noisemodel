import dm_env
from dm_env import specs
import numpy as np
from qibo import gates
from qibo.models import Circuit
from copy import deepcopy
from rlnoise.utils import kld


class CircuitsAcme(dm_env.Environment):

    def __init__(self, circuits_repr, labels):
        self.actions=(0,1)
        self.labels=labels
        self.len = len(circuits_repr[0])
        self.shape = np.shape(circuits_repr[0])
        self.circuits_repr = circuits_repr
        self._reset_next_step = True

    def reset(self) -> dm_env.TimeStep:
        self.position = 0
        self.sample = np.random.randint(low=0, high=len(self.circuits_repr))
        self.circuit = self.circuits_repr[self.sample]
        self.noisy_channels = np.zeros((self.len))
        self.observation_space = np.zeros((self.len, 4), dtype=np.float32)
        self.observation_space[:,0:2] = self.circuit
        self._reset_next_step = False
        return dm_env.restart(self._observation())

    def step(self, action: int) -> dm_env.TimeStep:
        self.last_action=action
        if self._reset_next_step:
            return self.reset()
        # place noisy gate
        if action == 1:
            self.noisy_channels[self.position]=1.
        # Check for termination.
        if self.position == (self.len-1):
            # Compute reward here
            reward = self.compute_reward(self.labels[self.sample], n_shots=100)
            self._reset_next_step = True
            return dm_env.termination(reward=reward, observation=self._observation())
        else:
            self.position+=1
            return dm_env.transition(reward=0., observation=self._observation())

    def compute_reward(self, label, n_shots=100):
        reward=0.
        generated_circuit = self.generate_circuit()
        observables = np.ndarray((3,2), dtype=float)
        index=0
        for obs in ["Z", "Y", "X"]:
            moments=self.pauli_probabilities(generated_circuit, obs, n_shots=n_shots)
            observables[index, :]=moments
            index+=1
        for i in range(3):
            reward+=kld(m1=observables[i,0], m2=label[i,0], v1=observables[i,1], v2=label[i,1])
        return reward

    def generate_circuit(self, dep_error=0.05):
      qibo_circuit = Circuit(1, density_matrix=True)
      for i in range(self.len):
        if self.circuit[i,0]==0:
          qibo_circuit.add(gates.RZ(0, theta=self.circuit[i,1]*2*np.pi, trainable=False))
        else:
          qibo_circuit.add(gates.RX(0, theta=self.circuit[i,1]*2*np.pi, trainable=False))
        if self.noisy_channels[i]==1:
          qibo_circuit.add(gates.DepolarizingChannel((0,), lam=dep_error))
      return qibo_circuit

    def pauli_probabilities(self, circuit, observable, n_shots=100, n_rounds=100):
        measured_circuit = deepcopy(circuit)
        self.add_masurement_gates(measured_circuit, observable=observable)
        register=np.ndarray((n_rounds,), dtype=float)
        moments=np.ndarray((2,), dtype=float)
        for i in range(n_rounds):
            probs=self.compute_shots(measured_circuit, n_shots=n_shots)
            register[i]=probs[0]-probs[1]
        moments[0]=np.mean(register)
        moments[1]=np.var(register)
        return moments

    def add_masurement_gates(self, circuit, observable):
        if observable=='X' or observable=='Y':
            circuit.add(gates.H(0))
        if observable=='Y':
            circuit.add(gates.SDG(0))
        circuit.add(gates.M(0))
        
    def compute_shots(self, circuit, n_shots=1024):
        shots_register_raw = circuit(nshots=n_shots).frequencies(binary=False)
        shots_register=tuple(int(shots_register_raw[key]) for key in range(2))
        return np.asarray(shots_register, dtype=float)/float(n_shots)

    def observation_spec(self) -> specs.BoundedArray:
        """Returns the observation spec."""
        return specs.BoundedArray(
            shape=self.observation_space.shape,
            dtype=self.observation_space.dtype,
            name="observation_space",
            minimum=0,
            maximum=1,
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec."""
        return specs.DiscreteArray(
            dtype=int, 
            num_values=len(self.actions), 
            name="action"
        )

    def _observation(self) -> np.ndarray:
        self.observation_space[:,3].fill(0.)
        self.observation_space[self.position,3] = 1.
        self.observation_space[:,2] = self.noisy_channels
        return self.observation_space.copy()

    def get_info(self):
      print("Circuit number: ", self.sample)
      if self._reset_next_step:
        print("Episode ended")
      else:
        print("Action number: ", self.position)
      print("Last action: ", self.last_action)
      #print(self.observation_spec)
      print("Observation: ", self.last_action)
      print(self._observation())