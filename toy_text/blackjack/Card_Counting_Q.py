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

  if s4 <= -10:
    s4 = 0
  elif s4 <= -5:
    s4 = 1
  elif -5 < s4 < 5:
    s4 = 2
  elif 5 <= s4 < 10:
    s4 = 3 
  else:
    s4 = 4

  output = s1
  output *= 11
  output += s2
  output *= 5
  output += s4
  output *= 2
  output += s3
  return output 

class Deck:
    def __init__(self):
        self.cards = [11,2,3,4,5,6,7,8,9,10,10,10,10]*24
        random.shuffle(self.cards)
        self.count = 0
        self.TC = 0

    def shuff(self):
        random.shuffle(self.cards)

    def draw(self):
        try:
            c = self.cards.pop(0)
        except:
            self.cards = [11,2,3,4,5,6,7,8,9,10,10,10,10]*24
            self.shuff()
            c = self.cards.pop(0)
        if 2 <= c <= 6:
            self.count += 1
        elif c == 11 or c == 10:
            self.count -= 1

        self.TC = self.count // (1+(len(self.cards))//52)
        return c


class Game:

    def hit(self, player, deck, dealer, s3):
        player.append(deck.draw())
        if sum(player) > 21:
            if s3 == 1:
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
        return player, reward, complete

    def stand(self, player, dealer, deck):
        while sum(dealer) < 17:
            dealer.append(deck.draw())
        if sum(dealer) > 21:
            if 11 in dealer:
                dealer[dealer.index(11)] = 1
                while sum(dealer) < 17:
                    dealer.append(deck.draw())
                if sum(dealer) > 21:
                    reward = 1
                elif sum(dealer) < sum(player):
                    reward = 1
                elif sum(dealer) == sum(player):
                    reward = 0
                else:
                    reward = -1
            else:
                reward = 1
        elif sum(dealer) < sum(player):
            reward = 1
        elif sum(dealer) == sum(player):
            reward = 0
        else:
            reward = -1

        complete = 1

        return reward, complete

    def start(self, deck):
        # Deal in Dealer
        dealer = []
        dealer.append(deck.draw())
        dealer.append(deck.draw())

        # Deal in Player
        player = []
        player.append(deck.draw())
        player.append(deck.draw())

        s1 = sum(player)
        s2 = dealer[1]
        s3 = 1 if 11 in player else 0
        s4 = deck.TC

        return player, dealer, s1, s2, s3, s4



def Q_learn(gamma, alpha, epsilon, n_episodes, decay, deck):
    """
    gamma: Discount rate
    alpha: Learning rate
    epsilon: Exploration rate
    n_episodes: Number of training episodes
    decay: Epsilon decay rate
    """
    max_steps = 500

    game = Game()
    
    Q = np.zeros((32 * 12 * 2 * 5, 2))
    for ep in range(n_episodes):

        player, dealer, s1, s2, s3, s4 = game.start(deck)

        state = encodeState(s1,s2,s3,s4)
        
        if not ep%10000:
            print(ep, "epsilon: {}".format(epsilon))
        
        #if np.random.uniform(0,1) < epsilon:
        #    action =  random.randint(0, 1) # take random action
        #else:
        #    action = np.argmax(Q[state])
        #print(deck.TC)

        i = 0
        while i < max_steps:

            if np.random.uniform(0,1) < epsilon:
                action = random.randint(0, 1) # take random action
            else:
                action = np.argmax(Q[state])
            
            if action == 1:
                player, reward, complete = game.hit(player, deck, dealer, s3)
                s1 = sum(player)

                s3 = 1 if 11 in player else 0
            else:
                reward, complete = game.stand(player, dealer, deck)

            s4 = deck.TC

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
    wins = 0
    losses = 0
    draws = 0
    deck = Deck()
    game = Game()
    over5 = 0
    under5 = 0
    o5w = 0
    u5w = 0
    for i in range(iterations):

        player, dealer, s1, s2, s3, s4 = game.start(deck)

        s = encodeState(s1,s2,s3,s4)

        complete = 0
        if not i % 10000:
            print(i, "######################")
        while not complete:
            # Choose action from Q table
            a = np.argmax(q[s])
            #if not i % 1000:
            #    print(s1,s2,s3,s4)

            if a == 1:
                player, reward, complete = game.hit(player, deck, dealer, s3)
                s1 = sum(player)

                s3 = 1 if 11 in player else 0
            else:
                reward, complete = game.stand(player, dealer, deck)
            #d = complete

            # Get new state & reward from environment
            #s1,r,d,_ = env.step(a)
            if complete:
                if deck.TC >= 0:
                    over5 += 1
                    o5w += reward
                elif deck.TC <= 0:
                    under5 += 1
                    u5w += reward
            #if not i % 1000:
            #    print(reward)
            if complete:
                if reward == 1:
                    wins += 1
                    #wins += max(1,deck.TC)
                elif reward == 0:
                    draws += 1
                else:
                    losses += 1
                    #losses += max(1,deck.TC)
                #if deck.TC >= 12 and reward == 1:
                #    o5w += 1
                #elif deck.TC <= -12 and reward == 1:
                #    u5w += 1

            s4 = deck.TC

            #s = s1
            s = encodeState(s1,s2,s3,s4)
    print(f"The agents average score was {((wins-losses)/iterations)}, and won {wins} times, lost {losses} times, drawed {draws} times")
    print(f"Over = {o5w/over5}", f"Under = {u5w/under5}")
    print(f"{o5w}, {over5}, {u5w}, {under5}")


play(1.0, 0.1, 1, 600000, 0.999998, 500000)


