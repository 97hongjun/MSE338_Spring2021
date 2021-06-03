import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import sys
import time
from mdp import *
from algorithms import *
from qlearning import *
from plot import *


if __name__ == "__main__":
    """
        0: 0 for discounted, 1 for differential q learning, 2 for shi q learning.
        1: epsilon_greedy parameter.
        2: eval period.
        3: T.
        4: alpha.
        5: seed.
        6: gamma if discounted.
    """
    args = sys.argv[1:]
    algorithm_version = int(args[0])
    epsilon_greedy = float(args[1])
    eval_period = int(args[2])
    T = int(args[3])
    alpha = float(args[4])
    seed = int(args[5])
    if algorithm_version == 0:
        gamma = float(args[6])

    depth = 1
    A = 11
    A_prime = np.floor((A-1) / 2)
    branch_factor = A - A_prime - 1
    D = 21
    parameters = (D, A, mdp_epsilon)
    eta = 1.0
    S = 0
    for i in range(depth+1):
        S += branch_factor**i
    T_epsilon = D*A*S
    mdp_epsilon = 0.2*np.sqrt((A-1)/(A*(D**2)))
    np.random.seed(seed)

    env_regret_arrs = []
    if algorithm_version == 1:
        for t in range(10):
            s_time = time.time()
            mdp = MDP(depth=depth, parameters=parameters)
            lambda_star = mdp.compute_lambda_star()
            np.random.seed(seed*2)
            regret_arrs  = []
            for i in range(10):
                regret_arr = differentialQ_learning(T, mdp, alpha, eta, version=1, epsilon=epsilon_greedy,
                                                    eval_period=eval_period, eval_steps=1000,
                                                    lambda_star=lambda_star)
                regret_arrs.append(regret_arr)
            env_regret_arrs.append(regret_arrs)
            e_time = time.time()
            print("Finished Iteration: %s, Time Taken: %s"%(t, e_time-s_time))
        env_regret_arrs = np.array(env_regret_arrs)
        filename = "diff_q_learning_regrets_epsilon%s_evalperiod%s_T%s_alpha%s_seed%s"%(epsilon_greedy, eval_period, T, alpha, seed)
        pklname = "pickle_data/%s.pkl"%filename
        with open(pklname, 'wb') as handle:
            pickle.dump(env_regret_arrs, handle)
        avg_regrets = np.sum(np.sum(env_regret_arrs, axis=0), axis=0)/100.0
        plot_regret(avg_regrets, eval_period, filename, algorithm_version)
    elif algorithm_version == 0:
        for t in range(10):
            s_time = time.time()
            mdp = MDP(depth=depth, parameters=parameters)
            lambda_star = mdp.compute_lambda_star()
            np.random.seed(seed*2)
            regret_arrs  = []
            for i in range(10):
                regret_arr = Q_learning(T, mdp, gamma, alpha, version=1, epsilon=epsilon_greedy,
                                        eval_period=eval_period, eval_steps=1000, lambda_star=lambda_star)
                regret_arrs.append(regret_arr)
            env_regret_arrs.append(regret_arrs)
            e_time = time.time()
            print("Finished Iteration: %s, Time Taken: %s"%(t, e_time-s_time))
        env_regret_arrs = np.array(env_regret_arrs)
        filename = "q_learning_regrets_epsilon%s_evalperiod%s_T%s_alpha%s_seed%s_gamma%s"%(epsilon_greedy, eval_period, T, alpha, seed, gamma)
        pklname = "pickle_data/%s.pkl"%filename
        with open(pklname, 'wb') as handle:
            pickle.dump(env_regret_arrs, handle)
        avg_regrets = np.sum(np.sum(env_regret_arrs, axis=0), axis=0)/100.0
        plot_regret(avg_regrets, eval_period, filename, algorithm_version)
    else:
        print("")
