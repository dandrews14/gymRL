# OpenAI's Taxi Environment

Agent tries to win a hand of blackjack. (Optimal strategy produces win percentage of [42.22%](https://betandbeat.com/blackjack/blog/is-blackjack-fair/))

Rewards: -1 for losing hand/bust, 0 for push, +1 for winning hand, +1.5 for natural blackjack.

Dealer hits until 17 or greater.

Tested on 100,000 iterations.

SARSA Results:
- Wins 38.059% of the time
- Average score of -0.15732

Q-learning Results:
- Wins 39.015% of the time
- Average score of -0.14859
