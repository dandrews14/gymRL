import math
import numpy as np
import gym
import colorama
import time
import sys
import random

def encodeState(s1,s2,s3,s4):
  #print(s1,s2,s3,s4)
  if s3:
      s3 = 1
  else:
      s3 = 0
  output = s1
  output *= 21
  output += s4
  output *= 11
  output += s2
  output *= 2
  output += s3
  return output 

class Deck:
    def __init__(self):
        self.cards = [11,2,3,4,5,6,7,8,9,10,10,10,10]*24
        self.count = 10

    def shuff(self):
        random.shuffle(self.cards)

    def draw(self):
        try:
            c = cards.pop()
        except:
            self.cards = [11,2,3,4,5,6,7,8,9,10,10,10,10]*24
            self.shuff()
            c = self.cards.pop()
        if 2 <= c <= 6:
            self.count += 1
        elif c == 11 or c == 10:
            self.count -= 1
        self.count = max(0,self.count)
        self.count = min(20,self.count)
        return c



def Q_learn(gamma, alpha, epsilon, n_episodes, decay, deck):
    """
    gamma: Discount rate
    alpha: Learning rate
    epsilon: Exploration rate
    n_episodes: Number of training episodes
    decay: Epsilon decay rate
    """
    max_steps = 500
    
    Q = np.zeros((32 * 11 * 2 * 22, 2))
    for ep in range(n_episodes):

        # Deal in Dealer
        dealer = []
        dealer.append(deck.draw())
        dealer.append(deck.draw())

        s2 = dealer[1]

        # Deal in Player
        player = []
        player.append(deck.draw())
        player.append(deck.draw())

        s1 = sum(player)

        s3 = 1 if 11 in player else 0

        s4 = deck.count

        state = encodeState(s1,s2,s3,s4)
        
        if not ep%10000:
            print(ep, "epsilon: {}".format(epsilon))
        
        if np.random.uniform(0,1) < epsilon:
            action =  random.randint(0, 1) # take random action
        else:
            action = np.argmax(Q[state])

        i = 0
        while i < max_steps:

            if np.random.uniform(0,1) < epsilon:
                action = random.randint(0, 1) # take random action
            else:
                action = np.argmax(Q[state])
            
            if action == 1:
                player.append(deck.draw())
                if sum(player) > 21:
                    if s3:
                        player[player.index(11)] = 1
                        complete = 0
                        reward = 0
                        s3 = 0
                    else:
                        complete = 1
                        reward = -1
                elif sum(player) == 21:
                    if sum(dealer) != 21:
                        reward = 1
                    else:
                        reward = 0
                    complete = 1
                else:
                    reward = 0
                    complete = 0
            else:
                while sum(dealer) < 17:
                    dealer.append(deck.draw())
                if sum(dealer) > 21 or sum(dealer) < sum(player):
                    reward = 1
                elif sum(dealer) == sum(player):
                    reward = 0
                else:
                    reward = -1

                complete = 1

            # Get next state
            #state2, reward, complete, _ = env.step(action)
            s1 = sum(player)

            s3 = 1 if 11 in player else 0

            s4 = deck.count


            state2 = encodeState(s1,s2,s3,s4)
            
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
    q = Q_learn(gamma, alpha, epsilon, n_episodes, decay, Deck())
    env = gym.make('Blackjack-v0')
    score = 0
    tot = 0
    deck = Deck()
    for i in range(iterations):
        #s = env.reset()

        # Deal in Dealer
        dealer = []
        dealer.append(deck.draw())
        dealer.append(deck.draw())

        s2 = dealer[1]

        # Deal in Player
        player = []
        player.append(deck.draw())
        player.append(deck.draw())

        s1 = sum(player)

        s3 = 1 if 11 in player else 0

        s4 = deck.count

        s = encodeState(s1,s2,s3,s4)

        #s = encodeState(s[0],s[1],s[2])
        d = False
        if not i % 1000:
            print(i, "######################")
        while not d:
            # Choose action from Q table
            a = np.argmax(q[s])
            if not i % 1000:
                print(s1,s2,s3,s4)

            if a == 1:
                player.append(deck.draw())
                if sum(player) > 21:
                    if s3:
                        player[player.index(11)] = 1
                        complete = 0
                        reward = 0
                        s3 = 0
                    else:
                        complete = 1
                        reward = -1
                elif sum(player) == 21:
                    if sum(dealer) != 21:
                        reward = 1
                    else:
                        reward = 0
                    complete = 1
                else:
                    complete = 0
            else:
                while sum(dealer) < 17:
                    dealer.append(deck.draw())
                if sum(dealer) > 21 or sum(dealer) < sum(player):
                    reward = 1
                elif sum(dealer) == sum(player):
                    reward = 0
                else:
                    reward = -1

                complete = 1
            d = complete
            # Get new state & reward from environment
            #s1,r,d,_ = env.step(a)
            if reward >= 1.0:
              tot += 1
            score += reward
            # Set new state

            s1 = sum(player)

            s3 = 1 if 11 in player else 0

            s4 = deck.count

            #s = s1
            s = encodeState(s1,s2,s3,s4)
    print("The agents average score was {}, and won {} percent of the time".format((score/iterations), (tot/iterations)*100))


play(1.0, 0.1, 1, 100000, 0.999998, 100000)