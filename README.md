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
- [gym](https://www.gymlibrary.ml/)
- [colorama](https://pypi.org/project/colorama/)

OpenAI Gym games solved:
- Toy text:
    - [Taxi](https://www.gymlibrary.ml/environments/toy_text/taxi/): SARSA, Q-learning
    - [Frozen Lake](https://www.gymlibrary.ml/environments/toy_text/frozen_lake/): SARSA, Q-learning
    - [Blackjack](https://www.gymlibrary.ml/environments/toy_text/blackjack/): SARSA, Q-learning

- Classic Control:
    - [Cart Pole](https://www.gymlibrary.ml/environments/classic_control/cart_pole/): REINFORCE
    - [Mountain Car](https://www.gymlibrary.ml/environments/classic_control/mountain_car/): REINFORCE

- Box2D
    - [Lunar Lander](https://www.gymlibrary.ml/environments/box2d/lunar_lander/): REINFORCE
