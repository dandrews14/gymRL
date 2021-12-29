import numpy as np
import matplotlib.pyplot as plt
import gym
import sys

import torch
from torch import nn
from torch import optim

print(sys.version)
print(torch.__version__)
print(torch.version.cuda)

class policy_estimator():
    def __init__(self, env):
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        
        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 16), 
            nn.ReLU(), 
            nn.Linear(16, self.n_outputs),
            nn.Softmax(dim=-1))
    
    def predict(self, state):
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs


env = gym.make('CartPole-v1')
s = env.reset()
pe = policy_estimator(env)
print(pe.predict(s))
print(pe.network(torch.FloatTensor(s)))


def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i] 
                  for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


def reinforce(env, policy_estimator, num_episodes=2000,
              batch_size=10, gamma=0.99):

    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1
    
    # Define optimizer
    optimizer = optim.Adam(policy_estimator.network.parameters(), 
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
            action_probs = policy_estimator.predict(s_0).detach().numpy()
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
                        policy_estimator.predict(state_tensor))
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
                
    return total_rewards


#rewards = reinforce(env, pe)
#window = 10
#smoothed_rewards = [np.mean(rewards[i-window:i+1]) if i > window 
#                    else np.mean(rewards[:i+1]) for i in range(len(rewards))]


def simulator():

    reward = 0
    env = gym.make('CartPole-v1')
    s_0 = env.reset()
    pe = policy_estimator(env)
    rewards = reinforce(env, pe)
    action_space = np.arange(env.action_space.n)  

    for i in range(100):
        s_0 = env.reset()

        while True:
            action_probs = pe.predict(s_0).detach().numpy()
            action = np.random.choice(action_space, p=action_probs)
            s_1, r, complete, _ = env.step(action)
            reward += r
            if complete:
                break
            if i >= 95:
                env.render()
            s_0 = s_1

    env.close()
    print("")
    print(reward/100)

simulator()
