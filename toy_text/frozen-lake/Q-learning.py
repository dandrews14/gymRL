import math
import numpy as np
import gym
import colorama
import time
import sys

colorama.init()


def Q_learn(gamma, alpha, epsilon, n_episodes, decay):
    """
    gamma: Discount rate
    alpha: Learning rate
    epsilon: Exploration rate
    n_episodes: Number of training episodes
    decay: Epsilon decay rate
    """
    env = gym.make('FrozenLake-v0')
    env.render()
    max_steps = 500
    
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    for ep in range(n_episodes):
        state = env.reset()
        if not ep%1000:
            print(ep, "epsilon: {}".format(epsilon))
        
        if np.random.uniform(0,1) < epsilon:
            action = env.action_space.sample() # take random action
        else:
            action = np.argmax(Q[state])

        i = 0
        while i < max_steps:

            if np.random.uniform(0,1) < epsilon:
                action = env.action_space.sample() # take random action
            else:
                action = np.argmax(Q[state])
            
            # Get next state
            state2, reward, complete, _ = env.step(action)
            
            # Update Q
            Q[state][action] = Q[state][action] + alpha*(reward + gamma*np.max(Q[state2]) - Q[state][action])

            if complete:
                epsilon = epsilon*decay
                break
            
            state = state2
            i += 1

    return Q

def play(gamma, alpha, epsilon, n_episodes, decay, iterations):
    """
    gamma: Discount rate
    alpha: Learning rate
    epsilon: Exploration rate
    n_episodes: Number of training episodes
    decay: Epsilon decay rate
    iterations: Number of testing iterations
    """
    q = Q_learn(gamma, alpha, epsilon, n_episodes, decay)
    env = gym.make('FrozenLake-v0')
    score = 0
    tot = 0
    for i in range(iterations):
        print("Round {}:".format(i))
        s = env.reset()
        d = False
        while d != True:
            #env.render()
            #time.sleep(0.5)

            # Choose action from Q table
            a = np.argmax(q[s,:])
            # Get new state & reward from environment
            s1,r,d,_ = env.step(a)
            if r == 1:
                score += 1
            tot += r
            # Set new state
            s = s1
        print("{}\n".format(r))
    print("The agent reached the end {} percent of the time, with an average score of {}".format((score/iterations)*100, tot/iterations))


play(0.99, 0.1, 1, 200000, 0.99998, 100)