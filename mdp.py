import numpy as np


class MDP():

  def __init__(self, depth, parameters):

    self.D, self.A, self.epsilon = parameters
    self.A_prime = np.floor((self.A-1)/2)
    self.delta = 4.0/self.D
    # print('Delta: {}'.format(self.delta))
    self.branch_factor = self.A - self.A_prime - 1

    ary_len = 0.0
    ary_len_penult = 0.0
    for i in range(depth):
      ary_len_penult += self.branch_factor**i
    for i in range(depth + 1):
      ary_len += self.branch_factor**i

    self.ary_len = int(ary_len)
    self.ary_len_penult = int(ary_len_penult)
    self.num_states = 2 * self.ary_len
    s1s = np.array(ary_len)
    s2s = np.array(ary_len)
    states = np.vstack([s1s, s2s])
    self.states = states
    self.s_state = np.random.randint(0, self.ary_len)

  def transition(self, state, action, print_=False):
    """
      action 0 is "good" action
    """
    assert action < self.A

    if action < self.A_prime or state[0] == 1:
      if state[0] == 0:
        if action == 0 and state[1]==self.s_state:
          if print_:
            print("optimal action selected")
          s0 = np.random.choice(2, p=[1.0-self.delta-self.epsilon, self.delta + self.epsilon])
        else:
          s0 = np.random.choice(2, p=[1.0-self.delta, self.delta])
        next_state = np.array([s0, state[1]])
      else:
        s0 = np.random.choice(2, p=[self.delta, 1.0-self.delta])
        next_state = np.array([s0, state[1]])
    else:
      #A_prime is the up action
      if state[1] > 0 and action == self.A_prime:
        s1 = np.floor((state[1]-1)/self.branch_factor)
        next_state = np.array([state[0], s1])
      elif action == self.A_prime:
        next_state = state
      elif state[1] < self.ary_len_penult:
        s1 = self.branch_factor*state[1] + (action - self.A_prime)
        # print(s1)
        next_state = np.array([state[0], s1])
      else:
        next_state = state

    return next_state

  def reward(self, curr_state, next_state):
    # Note(Saurabh): I changed this from next_state to curr_state
    # based on the MDP description in the paper.
    if curr_state[0] == 1:
      return 1.0
    return 0.0

  def compute_lambda_star(self):
    optimal_avg_return = (self.delta + self.epsilon)/(2.0*self.delta + self.epsilon)
    return optimal_avg_return
