# OpenAI's Lunar Lander Environment

Agent tries to Land a Spacecraft.

Rewards:
- 100-140 points for moving towards landing pad. 
- Loses points for moving away
- -100 for crashing
- +100 for landing softly
- +10 for each leg that touches the landing pad
- Firing main engine is -0.3
- Firing side enginer is -0.3
- Solved is 200 points

REINFORCE Results:
- Solved using ~3,000 episodes
- Average score of 500.0 over 1,000 episodes