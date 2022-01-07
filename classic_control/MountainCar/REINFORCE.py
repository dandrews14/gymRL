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

def discount_rewards(rewards, gamma):
    # Calculate rewards at each time step with discount
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])

    # Calculate cumulative sums for rewards
    r = r[::-1].cumsum()[::-1]

    # Normalize rewards
    return r - r.mean()

def calculate_loss(pol_est, states, actions, rewards):
    # Calculate log probabilities for actions
    act_probs = pred(pol_est, states)
    log_prob = torch.log(act_probs)

    # Multiply rewards by the log probabilities of the actions taken by the model
    selected_logprobs = rewards * log_prob[np.arange(len(actions)), actions]
    
    # Looking for local maximum
    # See ~32:30 mark of https://www.youtube.com/watch?v=bRfUxQs6xIM&list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb&index=6
    loss = -selected_logprobs.mean()

    return loss

def reinforce(env, policy_estimator, num_episodes=5000,
              batch_size=10, gamma=0.99):

    # Set up batches to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1
    
    # Define optimizer
    optimizer = optim.Adam(policy_estimator.parameters(), 
                           lr=0.01)
    
    # Get number of actions
    num_actions = env.action_space.n

    # Main loop (Runs for num_episodes or until the model converges on a good answer)
    for ep in range(num_episodes):

        # Initialize environment
        s_0 = env.reset()

        # Initialize replay buffers for episode
        states = []
        rewards = []
        actions = []
        complete = False

        # If model has converged, stop training
        #if np.mean(total_rewards[-250:]) >= 500.0:
        #  break

        # Run episode
        while complete == False:

            # Get action probabilities from model and convert to numpy array
            action_probs = pred(policy_estimator,(s_0)).detach().numpy()

            # Choose action
            action = np.random.choice(num_actions, p=action_probs)

            # Update environment
            s_1, r, complete, _ = env.step(action)

            r = abs(s_1[0]+0.5)
            env.render()
            #print(s_1)
            # Add state, reward, action to buffer
            states.append(s_0)
            rewards.append(r)
            actions.append(action)

            # Update state
            s_0 = s_1
            
            # If complete, batch data
            if complete:

                # Append all states, rewards, actions to batch
                batch_rewards.extend(discount_rewards(rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))
                
                # If batch is complete, update network
                if batch_counter == batch_size:

                    # Reset Gradients
                    optimizer.zero_grad()

                    # Convert batches to tensors
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(batch_rewards)

                    # Actions are used as indices, must be LongTensor
                    action_tensor = torch.LongTensor(batch_actions)
                    
                    # Calculate loss
                    loss = calculate_loss(policy_estimator, state_tensor, action_tensor, reward_tensor)
                    
                    # Calculate gradients
                    loss.backward()

                    # Apply gradients
                    optimizer.step()
                    
                    # Reset batches
                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1
                    
                # Print running average
                print("\rEp: {} Average of last 250: {:.2f}".format(
                    ep + 1, np.mean(total_rewards[-250:])), end="")
                
    return

def simulator(num_sims):
    # Total Reward Tracker
    reward = 0

    # Initialize environment
    env = gym.make('MountainCar-v0')

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
    num_actions = env.action_space.n

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
            action = np.random.choice(num_actions, p=action_probs)

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


if __name__ == "__main__":
    simulator(1000)
