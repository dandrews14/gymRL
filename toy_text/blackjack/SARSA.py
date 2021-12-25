import math
import numpy as np
import gym
import colorama
import time

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
    
def sarsa(gamma, alpha, epsilon, n_episodes, decay):
    """
    gamma: Discount rate
    alpha: Learning rate
    epsilon: Exploration rate
    n_episodes: Number of training episodes
    decay: Epsilon decay rate
    """
    env = gym.make('Blackjack-v0')

    Q = np.zeros((env.observation_space[0].n * env.observation_space[1].n * env.observation_space[2].n, env.action_space.n))
    
    for ep in range(n_episodes):
        state1 = env.reset()
        state1 = encodeState(state1[0], state1[1], state1[2])
        if not ep%10000:
            print(ep, "epsilon: {}".format(epsilon))

        if np.random.uniform(0,1) < epsilon:
            action1 = env.action_space.sample() # take random action
        else:
            action1 = np.argmax(Q[state1])

        while True:

            # Get next state
            state2, reward, complete, _ = env.step(action1)
            state2 = encodeState(state2[0], state2[1], state2[2])
            
            # Choose next action
            if np.random.uniform(0,1) < epsilon:
                action2 = env.action_space.sample() # take random action
            else:
                action2 = np.argmax(Q[state2])
            
            # Update Q
            target = reward + gamma*Q[state2, action2]
            Q[state1, action1] = Q[state1,action1] + alpha*(target-Q[state1, action1])
            
            state1 = state2
            action1 = action2
            
            if complete:
                epsilon = epsilon*decay
                break   
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
    q = sarsa(gamma, alpha, epsilon, n_episodes, decay)
    env = gym.make('Blackjack-v0')
    score = 0
    tot = 0
    for i in range(iterations):
        #print("Round {}:".format(i))
        s = env.reset()
        s = encodeState(s[0],s[1],s[2])
        d = False
        #print(s)
        while d != True:
            #env.render()
            #time.sleep(0.5)
            # Choose action from Q table
            a = np.argmax(q[s])
            # Get new state & reward from environment
            s1,r,d,_ = env.step(a)
            #print(s1)
            if r >= 1.0:
              tot += 1
            score += r
            # Set new state
            s = s1
            s = encodeState(s[0],s[1],s[2])
        #print("{}\n".format(r))
    print("The agents average score was {}, and won {} percent of the time".format((score/iterations), (tot/iterations)*100))


play(1.0, 0.1, 1, 500000, 0.999998, 100000)