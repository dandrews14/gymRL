# A brief summary of what each RL algorithm does
​
- [SARSA/Q-learning](https://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html): 
    - Overview
        - Both algorithms initialize a Q-table for the agent and then use Epsilon-Greedy 
to balance exploration/exploitation as the agent learns.
        - The Q-table is updated using the reward, learning rate, gamma value, 
and the estimated value of the state the agent ends up in after taking a given action.
        - SARSA is on-policy while Q-learning is off-policy. This means that an 
agent that learns using SARSA will update its Q-table based off of its current the
true state value. While Q-learning updates its Q-table based off of the best next 
possible action.
    - Strengths/Weaknesses
        - Due to the On-Policy/Off-Policy differences between Q-learning and SARSA,
each method has its own strengths and weaknesses. Q-learning is often preferable 
because of its ability to find the optimal policy. The standard example used to display
this is the [cliff walking experiment](https://github.com/cvhu/CliffWalking).
    - Overall, both algorithms are good for simple, discrete state spaces. However,
they can struggle with larger and more complicated problems.
​
- [Vanilla Policy Gradient/REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html#background):
​
​
- [Deep Q-learning Network (DQN)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf):
​
​
- [Advantage Actor Critic (A2C)](https://arxiv.org/pdf/1602.01783.pdf):
​
​
- [Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html):
​
​
- [Deep Deterministic Policy Gradients (DDPG)](https://spinningup.openai.com/en/latest/algorithms/ddpg.html):