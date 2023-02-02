import dm_env
from dm_env import specs
import numpy as np

class CircuitEnv(dm_env.Environment):

    def __init__(self, circuit):
        self.actions=(0,1)
        self.len = len(circuit)
        self.shape = np.shape(circuit)
        self.circuit = circuit
        self.position = 0
        self.noisy_channels = np.zeros((len(circuit)))
        self.observation_space = np.zeros((len(circuit), 4), dtype=np.float)
        self._reset_next_step = True

    def reset(self) -> dm_env.TimeStep:
        self.position = 0
        self.observation_space.fill(0.)
        self.observation_space[:,0:2] = self.circuit
        self._reset_next_step = False
        self.noisy_channels.fill(0.)
        return dm_env.restart(self._observation())

    def step(self, action: int) -> dm_env.TimeStep:
        if self._reset_next_step:
            return self.reset()
        # place noisy gate
        if action == 1:
            self.noisy_channels[self.position]=1.
        self.position+=1
        # Check for termination.
        if self.position == (self.len):
            # Compute reward here
            reward = 1
            self._reset_next_step = True
            return dm_env.termination(reward=reward, observation=self._observation())
        else:
            return dm_env.transition(reward=0., observation=self._observation())

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
            dtype=int, num_values=len(self.actions), name="action")

    def _observation(self) -> np.ndarray:
        self.observation_space[:,3].fill(0.)
        self.observation_space[self.position,3] = 1.
        self.observation_space[:,2] = self.noisy_channels
        return self.observation_space.copy()


'''
class Catch(dm_env.Environment):
  """A Catch environment built on the `dm_env.Environment` class.
  The agent must move a paddle to intercept falling balls. Falling balls only
  move downwards on the column they are in.
  The observation is an array shape (rows, columns), with binary values:
  zero if a space is empty; 1 if it contains the paddle or a ball.
  The actions are discrete, and by default there are three available:
  stay, move left, and move right.
  The episode terminates when the ball reaches the bottom of the screen.
  """

  def __init__(self, rows: int = 10, columns: int = 5, seed: int = 1):
    """Initializes a new Catch environment.
    Args:
      rows: number of rows.
      columns: number of columns.
      seed: random seed for the RNG.
    """
    self._rows = rows
    self._columns = columns
    self._rng = np.random.RandomState(seed)
    self._board = np.zeros((rows, columns), dtype=np.float32)
    self._ball_x = None
    self._ball_y = None
    self._paddle_x = None
    self._paddle_y = self._rows - 1
    self._reset_next_step = True

  def reset(self) -> dm_env.TimeStep:
    """Returns the first `TimeStep` of a new episode."""
    self._reset_next_step = False
    self._ball_x = self._rng.randint(self._columns)
    self._ball_y = 0
    self._paddle_x = self._columns // 2
    return dm_env.restart(self._observation())

  def step(self, action: int) -> dm_env.TimeStep:
    """Updates the environment according to the action."""
    if self._reset_next_step:
      return self.reset()

    # Move the paddle.
    dx = _ACTIONS[action]
    self._paddle_x = np.clip(self._paddle_x + dx, 0, self._columns - 1)

    # Drop the ball.
    self._ball_y += 1

    # Check for termination.
    if self._ball_y == self._paddle_y:
      reward = 1. if self._paddle_x == self._ball_x else -1.
      self._reset_next_step = True
      return dm_env.termination(reward=reward, observation=self._observation())
    else:
      return dm_env.transition(reward=0., observation=self._observation())

  def observation_spec(self) -> specs.BoundedArray:
    """Returns the observation spec."""
    return specs.BoundedArray(
        shape=self._board.shape,
        dtype=self._board.dtype,
        name="board",
        minimum=0,
        maximum=1,
    )

  def action_spec(self) -> specs.DiscreteArray:
    """Returns the action spec."""
    return specs.DiscreteArray(
        dtype=int, num_values=len(_ACTIONS), name="action")

  def _observation(self) -> np.ndarray:
    self._board.fill(0.)
    self._board[self._ball_y, self._ball_x] = 1.
    self._board[self._paddle_y, self._paddle_x] = 1.
    return self._board.copy()
'''