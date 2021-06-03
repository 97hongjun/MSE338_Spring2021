import numpy as np
from algorithms import *

def differentialQ_learning(T, mdp, alpha, eta, version, epsilon=0.0,
                           eval_period=1000, eval_steps=1000, lambda_star=0):
  """
  Differential Q learning

  version:
    0: Greedy Q Learning
    1: Epsilon Greedy Q Learning
    2: Optimistic Q Learning
  """
  ### Optimistic differential Q Learning
  if version == 2:
    optimal_return = (mdp.delta + mdp.epsilon)/(2.0*mdp.delta + mdp.epsilon)
    Q_vals1 = np.ones((mdp.ary_len, mdp.A))*optimal_return*T
    Q_vals2 = np.ones((mdp.ary_len, mdp.A))*optimal_return*T
  else:
    Q_vals1 = np.zeros((mdp.ary_len, mdp.A))
    Q_vals2 = np.zeros((mdp.ary_len, mdp.A))

  cumulative_regret_arr = []
  policy_arr = []

  start = np.array([0,0])
  state = start
  avg_reward = 0.0
  for t in range(T):
    if t % eval_period == 0:
      regret = evaluate_policy(mdp, eval_steps, Q_vals1, Q_vals2, lambda_star=lambda_star, print_=False)
      cumulative_regret_arr.append(regret)

    """
      Indexing:
      state = [x, y].
      x in {0, 1} denoting if in s_0 or s_1.
      y in {0, 1, ..., ary_len} denoting which node of tree we are in.
    """
    a_t = policy(mdp, state, Q_vals1, Q_vals2, epsilon, version)
    s_t1 = mdp.transition(state, a_t)
    curr_reward = mdp.reward(state, s_t1)
    s_t1 = s_t1.astype(int)
    if s_t1[0] == 0:
      if state[0] == 0:
        delta = curr_reward - avg_reward + np.max(Q_vals1[s_t1[1]]) - Q_vals1[state[1], a_t]
        Q_vals1[state[1], a_t] = Q_vals1[state[1], a_t] + alpha*delta
      else:
        delta = curr_reward - avg_reward + np.max(Q_vals1[s_t1[1]]) - Q_vals2[state[1], a_t]
        Q_vals2[state[1], a_t] = Q_vals2[state[1], a_t] + alpha*delta
    else:
      if state[0] == 0:
        delta = curr_reward - avg_reward + np.max(Q_vals2[s_t1[1]]) - Q_vals1[state[1], a_t]
        Q_vals1[state[1], a_t] = Q_vals1[state[1], a_t] + alpha*delta
      else:
        delta = curr_reward - avg_reward + np.max(Q_vals2[s_t1[1]]) - Q_vals2[state[1], a_t]
        Q_vals2[state[1], a_t] = Q_vals2[state[1], a_t] + alpha*delta
    avg_reward = avg_reward + eta*alpha*delta
    state = s_t1
  return cumulative_regret_arr

def Q_learning(T, mdp, gamma, alpha, version, epsilon=0.0,
               eval_period=1000, eval_steps=1000, lambda_star=0):
  """
    Standard Q learning

    version:
      0: Greedy Q Learning
      1: Epsilon Greedy Q Learning
      2: Optimistic Q Learning
  """
  ### Optimistic Q Learning
  if version == 2:
    optimal_return = (mdp.delta + mdp.epsilon)/(2.0*mdp.delta + mdp.epsilon)
    Q_vals1 = np.ones((mdp.ary_len, mdp.A))*optimal_return
    Q_vals2 = np.ones((mdp.ary_len, mdp.A))*optimal_return
  else:
    Q_vals1 = np.zeros((mdp.ary_len, mdp.A))
    Q_vals2 = np.zeros((mdp.ary_len, mdp.A))

  start = np.array([0,0])
  state = start

  cumulative_regret_arr = []

  for t in range(T):
    if t % eval_period == 0:
      cumulative_regret_arr.append(
          evaluate_policy(mdp, eval_steps, Q_vals1, Q_vals2, lambda_star=lambda_star))

    a_t = policy(mdp, state, Q_vals1, Q_vals2, epsilon, version)
    s_t1 = mdp.transition(state, a_t)
    curr_reward = mdp.reward(state, s_t1)

    s_t1 = s_t1.astype(int)

    if s_t1[0] == 0:
      if state[0] == 0:
        delta = curr_reward + gamma*np.max(Q_vals1[s_t1[1]]) - Q_vals1[state[1], a_t]
        Q_vals1[state[1], a_t] = Q_vals1[state[1], a_t] + alpha*delta
      else:
        delta = curr_reward + gamma*np.max(Q_vals1[s_t1[1]]) - Q_vals2[state[1], a_t]
        Q_vals2[state[1], a_t] = Q_vals2[state[1], a_t] + alpha*delta
    else:
      if state[0] == 0:
        delta = curr_reward + gamma*np.max(Q_vals2[s_t1[1]]) - Q_vals1[state[1], a_t]
        Q_vals1[state[1], a_t] = Q_vals1[state[1], a_t] + alpha*delta
      else:
        delta = curr_reward + gamma*np.max(Q_vals2[s_t1[1]]) - Q_vals2[state[1], a_t]
        Q_vals2[state[1], a_t] = Q_vals2[state[1], a_t] + alpha*delta
    state = s_t1
  return cumulative_regret_arr
