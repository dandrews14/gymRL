import math
import numpy as np
import gym
import colorama
import time

colorama.init()

    
def sarsa(gamma, alpha, epsilon, n_episodes, decay):
    """
    gamma: Discount rate
    alpha: Learning rate
    epsilon: Exploration rate
    n_episodes: Number of training episodes
    decay: Epsilon decay rate
    """
    env = gym.make('Taxi-v3')
    env.render()
    
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    for ep in range(n_episodes):
        state1 = env.reset()
        
        if np.random.uniform(0,1) < epsilon:
            action1 = env.action_space.sample() # take random action
        else:
            action1 = np.argmax(Q[state1])

        while True:

            # Get next state
            state2, reward, complete, _ = env.step(action1)
            
            # Choose next action
            if np.random.uniform(0,1) < epsilon:
                action2 = np.random.randint(4) # take random action
            else:
                action2 = np.argmax(Q[state2, :])
            
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

# Train agent and test performance
play(0.95, 0.8, 1, 50000, 0.999, 100)