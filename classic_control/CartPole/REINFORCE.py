# https://towardsdatascience.com/learning-reinforcement-learning-reinforce-with-pytorch-5e8ad7fc7da0
# https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf

import numpy as np
import matplotlib.pyplot as plt
import gym

import torch
from torch import nn
from torch import optim

def pred(nw, state):
        # Use network and state to retrieve action probbilities
        action_probs = nw(torch.FloatTensor(state))
        return action_probs

def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i] 
                  for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


def reinforce(env, policy_estimator, num_episodes=3000,
              batch_size=10, gamma=0.99):

    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1
    
    # Define optimizer
    optimizer = optim.Adam(policy_estimator.parameters(), 
                           lr=0.01)
    
    action_space = np.arange(env.action_space.n)
    for ep in range(num_episodes):
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        complete = False
        while complete == False:
            # Get actions and convert to numpy array
            action_probs = pred(policy_estimator,(s_0)).detach().numpy()
            action = np.random.choice(action_space, p=action_probs)
            s_1, r, complete, _ = env.step(action)

            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1
            
            # If complete, batch data
            if complete:
                batch_rewards.extend(discount_rewards(rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))
                
                # If batch is complete, update network
                if batch_counter == batch_size:
                    optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(batch_rewards)
                    # Actions are used as indices, must be LongTensor
                    action_tensor = torch.LongTensor(batch_actions)
                    
                    # Calculate loss
                    logprob = torch.log(
                        pred(policy_estimator, state_tensor))
                    selected_logprobs = reward_tensor * \
                        logprob[np.arange(len(action_tensor)), action_tensor]
                    loss = -selected_logprobs.mean()
                    
                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    optimizer.step()
                    
                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1
                    
                # Print running average
                print("\rEp: {} Average of last 10: {:.2f}".format(
                    ep + 1, np.mean(total_rewards[-10:])), end="")
                
    return

def simulator(num_sims):
    # Total Reward Tracker
    reward = 0

    # Initialize environment
    env = gym.make('CartPole-v1')

    # Define NN Shape
    inputs = env.observation_space.shape[0]
    outputs = env.action_space.n
    pe = nn.Sequential(
            nn.Linear(inputs, 32), 
            nn.ReLU(), 
            nn.Linear(32, 16), 
            nn.ReLU(), 
            nn.Linear(16, outputs),
            nn.Softmax(dim=-1))

    # Train Neural Network
    reinforce(env, pe)
    action_space = np.arange(env.action_space.n)  

    # Run tests
    print("")
    for i in range(num_sims):

        # Get Initial State
        s_0 = env.reset()
        if i % 100 == 0:
          print(f"Round: {i}")


        while True:
            # Get Action Probabilities
            action_probs = pred(pe, s_0).detach().numpy()

            # Choose Action Nondeterministically
            action = np.random.choice(action_space, p=action_probs)

            # Take action
            s_1, r, complete, _ = env.step(action)

            # Add reward to total rewards
            reward += r

            # Check if simulations are complete
            if complete:
                break

            # Display last 5 simulations 
            if i >= num_sims-5:
                env.render()

            # Update State
            s_0 = s_1

    # Clean up and print results
    env.close()
    print("")
    print(f"Average score over 1000 rounds: {reward/num_sims}")

simulator(1000)
