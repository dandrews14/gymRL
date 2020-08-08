# A brief summary of what each RL algorithm does

- [SARSA/Q-learning](https://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html): 
    - Both algorithms initialize a Q-table for the agent and then use epsilon greedy to balance exploration/exploitation as the agent learns.
    - The Q-table is updated using the reward, and the estimated value of the state the agent ends up in after taking a given action.
    - SARSA is on policy while Q-learning is off policy and uses the max value of s' to update its Q-table.
    - Good for simple, discrete state spaces