import numpy as np
import matplotlib.pyplot as plt

def plot_regret(regret_values, eval_period, filename, algorithm_version):
  steps = np.arange(len(regret_values))*eval_period
  plt.plot(steps, np.cumsum(regret_values)*eval_period, color='blue', label='Per Timestep Regret')
  plt.xlabel('Training Steps')
  plt.ylabel('Average Cumulative Regret')
  if algorithm_version == 0:
      plt.title('Discounted Q Learning')
  else:
      plt.tilte('Differential Q Learning')
  plt.legend()
  plt.savefig("plots/%s.png"%filename, dpi=150)
