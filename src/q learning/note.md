# Q-Learning - Simple Revision Notes

## 1) What is Q-Learning?
Q-Learning is a reinforcement learning algorithm that learns how good an action `a` is in a state `s`.

It stores this as:
`Q(s, a)`

Higher `Q` means better long-term reward.

## 2) Meaning of Q(s, a)
`Q(s, a)` = expected total future reward if you:
- take action `a` in state `s`
- then follow the best policy afterward

It includes:
- immediate reward
- discounted future rewards

## 3) Core Update Rule (Most Important)
`Q(s,a) = Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))`

Simple meaning:
- New Q = Old Q + learning step * (new estimate - old Q)
- New estimate = reward + discounted best future value

This update is the learning process.

## 4) What is Learned?
Q-Learning learns a Q-table:
- rows: states
- columns: actions

Example (3x3 grid):
- 9 states
- 4 actions per state
- total 36 Q-values

After training:
- best action in each state = action with highest Q-value

## 5) How Optimal Path Appears
Optimal policy:
`pi(s) = argmax_a Q(s,a)`

Meaning:
- in each state, pick the action with highest Q

Q-Learning does not directly search for paths.
It learns values.
The path emerges from those values.

## 6) Hyperparameters
### epsilon (eps)
- exploration rate
- with probability `eps`: random action
- otherwise: best known action

### alpha
- learning rate
- high alpha: faster, less stable
- low alpha: slower, more stable

### gamma
- discount factor
- close to 1: long-term reward matters more
- close to 0: immediate reward matters more

### episodes
- number of training runs
- more episodes usually improves learning

## 7) Why It Works
With repeated updates, Q-values move toward optimal values:
`Q*(s,a) = r + gamma * max_a' Q*(s',a')`

In finite environments, this converges to the optimal solution (under standard conditions).

## 8) Key Reminders
- Q-Learning is model-free.
- It learns from experience.
- It balances exploration and exploitation.
- It learns values, not paths.
- Optimal path comes from picking highest-Q actions.

## One-Line Memory
Q-Learning learns the long-term value of each action in each state, then chooses the highest-value action.