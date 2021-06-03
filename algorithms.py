import numpy as np


# Epsilon-greedy policy
def policy(mdp, state, Q_vals1, Q_vals2, epsilon=0.0, version=0):
  if version == 1 or version == 2:
    rand_act = np.random.choice(2, p=[1.0-epsilon, epsilon])
    if rand_act:
      return np.random.randint(0, mdp.A)
  state = state.astype(int)
  if state[0] == 0:
    argmax_Qs = np.squeeze(np.argwhere(Q_vals1[state[1]] == np.amax(Q_vals1[state[1]])), axis=1)
    return np.random.choice(argmax_Qs)
  else:
    argmax_Qs = np.squeeze(np.argwhere(Q_vals2[state[1]] == np.amax(Q_vals2[state[1]])), axis=1)
    return np.random.choice(argmax_Qs)

def evaluate_policy(mdp, eval_steps, Q_vals1, Q_vals2, lambda_star=0, print_=False):
  # Compute the regret
  start = np.array([0,0])
  state = start
  cumulative_regret = 0.
  for t in range(eval_steps):
    # Note: the policy that is evaluated is the greedy policy.
    a_t = policy(mdp, state, Q_vals1, Q_vals2)
    s_t1 = mdp.transition(state, a_t, print_)
    curr_reward = mdp.reward(state, s_t1)
    if print_:
      print('State: {}, Action: {}, Next State: {}, Branch: {}, Reward: {}'.format(state, a_t, s_t1, s_t1[1], curr_reward))

    state = s_t1
    cumulative_regret += lambda_star - curr_reward

  return cumulative_regret / eval_steps

def check_policy(eval_steps, Q_vals1, Q_vals2, desired_state):
  # Compute the regret
  agmax = np.unravel_index(np.argmax(Q_vals1), Q_vals1.shape)
  if agmax[0] == desired_state and agmax[1] == 0:
    return 0.0
  return 1.0
