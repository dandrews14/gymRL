import math
import numpy as np
import gym
import colorama
import time

colorama.init()

    
def sarsa(gamma, alpha, epsilon, n_episodes, decay):
    env = gym.make('Taxi-v3')
    env.render()
    
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    for ep in range(n_episodes):
        state = env.reset()
        if not ep%1000:
            print(ep, "epsilon: {}".format(epsilon))
        
        if np.random.uniform(0,1) < epsilon:
            action = env.action_space.sample() # take random action
        else:
            action = np.argmax(Q[state])

        while True:

            # Get next state
            state2, reward, complete, _ = env.step(action)

            if reward == 20:
                Q[state][action] = Q[state][action] + alpha*(reward - Q[state][action])
                break

            elif complete:
                epsilon = epsilon*decay
                break
            
            # Update Q
            else:
                Q[state][action] = Q[state][action] + alpha*(reward + gamma*np.max(Q[state2]) - Q[state][action])
            
            state = state2
            
    return Q

def play(gamma, alpha, epsilon, n_episodes, decay, iterations):
    q = sarsa(gamma, alpha, epsilon, n_episodes, decay)
    env = gym.make('Taxi-v3')
    score = 0
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
            if r == 20:
                score += 1
            # Set new state
            s = s1
        print("{}\n".format(r))
    print("The agent had succesful dropoffs {} percent of the time".format((score/iterations)*100))

play(0.95, 0.8, 1, 100000, 0.9999, 100)