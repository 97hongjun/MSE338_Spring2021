# MSE338_Spring2021

**Running Code:**

```
python3 process.py algorithm_version epsilon_greedy_parameter eval_period T alpha seed gamma
```

**algorithm_version:** 0 for discounted, 1 for differential\
**epsilon_greedy_parameter:** epsilon in epsilon greedy\
**eval_period:** How frequently to evalulate policy\
**T:** total number of timesteps\
**alpha:** TD learning parameter\
**seed:** random seed\
**gamma:** if algorithm_version is 0, the discount factor, if 1 do not add this argument\

**Accessing Results:**

Pickle files of regret values are available under `pickle_data\`.\
Plots are available under `plots`.
