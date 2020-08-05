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
        state1 = env.reset()
        #print(reward)
        
        if np.random.uniform(0,1) < epsilon:
            action1 = np.random.randint(4)# take random action
        else:
            action1 = np.argmax(Q[state1, :])
        #print("######################")
        while True:
            #env.render()
            # Get next state
            state2, reward, complete, _ = env.step(action1)
            #print(reward)
            
            # Choose next action
            if np.random.uniform(0,1) < epsilon:
                action2 = np.random.randint(4) # take random action
            else:
                action2 = np.argmax(Q[state2, :])
            
            #Q = sarsa(state, state2, reward, action, action_2, gamma, alpha, epsilon, Q)
            # Update Q
            prediction = Q[state1, action1]
            target = reward + gamma*Q[state2, action2]
            Q[state1, action1] = Q[state1,action1] + alpha*(target-prediction)
            
            state1 = state2
            action1 = action2
            
            if complete:
                epsilon = epsilon*decay
                break
    #print(rewards)        
    return Q

def play(gamma, alpha, epsilon, n_episodes, decay):
    q = sarsa(gamma, alpha, epsilon, n_episodes, decay)
    env = gym.make('Taxi-v3')
    score = 0
    # Reset environment
    for i in range(100):
        print("Round {}:\n".format(i))
        s = env.reset()
        d = False
        # The Q-Table learning algorithm
        while d != True:
            #env.render()
            #time.sleep(0.5)
            # Choose action from Q table
            a = np.argmax(q[s,:])
            #Get new state & reward from environment
            s1,r,d,_ = env.step(a)
            if r == 20:
                score += 1

            #Update Q-Table with new knowledge
            s = s1
    print(score/100)

play(0.95, 0.8, 1, 50000, 0.99)