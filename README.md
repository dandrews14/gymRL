# An attempt at solving all of OpenAI's Gym environments from scratch using Reinforcement Learning

Algorithms used:
- [SARSA/Q-learning](https://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html)
- [Vanilla Policy Gradient/REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html#background)
- [Deep Q-learning Network (DQN)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- [Advantage Actor Critic (A2C)](https://arxiv.org/pdf/1602.01783.pdf)
- [Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Deep Deterministic Policy Gradients (DDPG)](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

Libraries used:
- [NumPy](https://numpy.org/)
- [gym](https://gym.openai.com/)
- [colorama](https://pypi.org/project/colorama/)

OpenAI Gym games solved:
- Toy text:
    - [Taxi](https://gym.openai.com/envs/Taxi-v3/): SARSA, Q-learning
    - [Frozen Lake](https://gym.openai.com/envs/FrozenLake-v0/): SARSA, Q-learning
    - [Blackjack](https://gym.openai.com/envs/Blackjack-v0/): SARSA, Q-learning

- Classic Control:
    - [Cart Pole](https://gym.openai.com/envs/CartPole-v1/): REINFORCE
    - [Mountain Car](https://gym.openai.com/envs/MountainCar-v0/): REINFORCE

- Box2D
    - [Lunar Lander](https://gym.openai.com/envs/LunarLander-v2/): REINFORCE
