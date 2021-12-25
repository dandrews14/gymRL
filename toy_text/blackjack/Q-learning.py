import math
import numpy as np
import gym
import colorama
import time
import sys

def encodeState(s1,s2,s3):
  if s3:
      s3 = 1
  else:
      s3 = 0
  output = s1
  output *= 11
  output += s2
  output *= 2
  output += s3
  return output 


def Q_learn(gamma, alpha, epsilon, n_episodes, decay):
    """
    gamma: Discount rate
    alpha: Learning rate
    epsilon: Exploration rate
    n_episodes: Number of training episodes
    decay: Epsilon decay rate
    """
    env = gym.make('Blackjack-v0')
    max_steps = 500
    
    Q = np.zeros((env.observation_space[0].n * env.observation_space[1].n * env.observation_space[2].n, env.action_space.n))

    for ep in range(n_episodes):
        state = env.reset()
        state = encodeState(state[0], state[1], state[2])
        
        if not ep%10000:
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
            
            state2 = encodeState(state2[0], state2[1], state2[2])
            
            # Update Q
            Q[state][action] = Q[state][action] + alpha*(reward + gamma*Q[state2][np.argmax(Q[state2])] - Q[state][action])

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
    np.set_printoptions(threshold=sys.maxsize)
    q = Q_learn(gamma, alpha, epsilon, n_episodes, decay)
    env = gym.make('Blackjack-v0')
    score = 0
    tot = 0
    for i in range(iterations):
        s = env.reset()
        s = encodeState(s[0],s[1],s[2])
        d = False
        while d != True:
            # Choose action from Q table
            a = np.argmax(q[s])
            # Get new state & reward from environment
            s1,r,d,_ = env.step(a)
            if r >= 1.0:
              tot += 1
            score += r
            # Set new state
            s = s1
            s = encodeState(s[0],s[1],s[2])
    print("The agents average score was {}, and won {} percent of the time".format((score/iterations), (tot/iterations)*100))


play(1.0, 0.1, 1, 500000, 0.999998, 100000)