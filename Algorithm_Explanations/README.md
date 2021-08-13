# A brief summary of what each RL algorithm does

## [SARSA/Q-learning](https://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html): 
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

## [Vanilla Policy Gradient/REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html#background):
- Overview
    - Policy Gradient algorithm.
    - Initializes a Neural Network and then repeatedly plays N steps of a game. Calculates the discounted reward and the expected reward. Back-propogate error to adjust
the weights for the NN to increase the expected reward.
- Strenths/Weaknesses
    - Update process is inefficient.
    - Algorithm can struggle with larger state spaces.
    - Fiarly simple implementation.


## [Deep Q-learning Network (DQN)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf):
- Overview
    - Similar to VPG, but the algorithm is focused on values instead of policies.
    - Initializes a neural network and replay buffer for training. Then repeatedly use Epsilon-greedy to select actions. Store <s,a,r,s'> in the replay buffer. 
Then we use random batches of transitions from the buffer to calculate the loss and update the network.
- Strenths/Weaknesses
    - Very strong algorithm.
    - Can be slow during training.


## [Actor Critic Models](https://arxiv.org/pdf/1602.01783.pdf):
- Overview
     - One of the newer Deep RL Frameworks.
     - Uses two Neural Networks. One Actor, one Critic.
     - Actor network is used to guide the agent. Outputs are probabilites for each potential action. The weights are updated after each time step.
     - Critic network maps state to Q-value. Updated after every time step.
     - Advanatage Actor Critic (A2C) uses something called advantage to help improve the model. Advantage essentially tells the model if a state is better
or worse than expected. It can help the model keep track of how much it is learning by visiting a given state.
     - A newer version of A2C exists called Asynchronous Advantage Actor Critic (A3C), although researchers at OpenAI have found no significant performance boost
from the newer algorithm.
- Strenths/Weaknesses
     - Another very strong algorithm.
     - Reduces training complexity using Advantage.
     - A2C and A3C have various trade-offs in terms of speed/complexity based off of what type of computer architecture is being used.
     - Implementation can be complicated.

## [Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html):
- Overview
     - OpenAI's prefered RL algorithm.
     - On-Policy Algorithm.
     - An extension of the [Trust Region Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/trpo.html) (TRPO) algorithm.
     - Looks for a way to speed up training by taking the largest step possible during updates.
     - Uses something called the [Kullback-Leibler Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) KL-Divergence. 
Which is used to measure the difference between two probability distributions. This metric is used in the "Penalty" version of the algorithm similar to that of TRPO.
The primary difference is that TRPO uses it as a contraint during updates, while PPO uses it to scale updates as a penalty.
     - The version of PPO that OpenAI uses is called the "Clip" version. In this algorithm, PPO uses a clipping function to persuade the algorithm to make
 smaller updates.
     - Overall very similar to VPG and A2C. Primary difference is the use of its clipping function to adjust the weight of each update.
- Strenths/Weaknesses
     - A fairly simple and successful algorithm.
     - Can occasionally get stuck in local optima.

## [Deep Deterministic Policy Gradients (DDPG)](https://spinningup.openai.com/en/latest/algorithms/ddpg.html):
- Overview
    - Off-Policy.
    - Essentially a combination of Actor Critic and DQN.
    - Here's a [good explanation of the differences](https://www.linkedin.com/pulse/ddpg-dqn-which-use-ridhwanul-haque)
- Strenths/Weaknesses
    - DDPG is generally better with continuous action spaces than DQN.