# DQN - Clean Notes (From Our Discussion)

## 1. Q-Learning Basics

- `Q` is a function of `(state, action)`.
- We choose action using:
  - `a = argmax_a Q(s, a)`
- Goal: maximize cumulative reward.

## 2. Why Tabular Q Fails

- Tabular Q-learning works only for small, discrete state spaces.
- CartPole has continuous states.
- Continuous values create effectively infinite combinations, so a Q-table is not practical.
- Solution: approximate `Q(s, a)` with a neural network.

## 3. CartPole Setup

- State has 4 values:
  - `cart_position`
  - `cart_velocity`
  - `pole_angle`
  - `pole_angular_velocity`
- Input layer size: `4`
- Actions:
  - `0` -> left
  - `1` -> right
- Output layer size: `2`
- Network output:
  - `[Q(s, left), Q(s, right)]`

## 4. Core DQN Flow

1. Get state:
   - `s = [0.10, 0.00, 0.02, 0.10]`
2. Predict Q-values:
   - `model(s) = [2.0, 3.4]`
3. Choose action:
   - `argmax -> action = 1 (right)`
4. Environment responds:
   - `s' = [0.20, 0.40, 0.03, 0.60]`
   - `reward = 1`
   - `done = False`

## 5. Learning Update

- If `done == False`:
  - `target = r + gamma * Q(next)`
- If `done == True`:
  - `target = r`

Loss:

- `(predicted - target)^2`

Backpropagation updates network weights.

## 6. Experience Replay

Store transitions:

- `(state, action, reward, next_state, done)`

Instead of learning from only the latest transition:

- Randomly sample a batch (for example, `32`)
- Compute targets
- Compute average loss
- Run one gradient update

Purpose:

- Break correlation between consecutive samples
- Stabilize learning
- Reuse old experience

## 7. Moving Target Problem

If the same network is used for both:

- selecting action
- computing target

then the target changes constantly during training, causing instability.

## 8. Target Network

Use two networks:

- Online model -> learns each step
- Target model -> frozen copy

Target equation:

- `target = r + gamma * Q_target(s')`

Update periodically:

- `target_model <- model`

Purpose:

- Stabilize training targets

## 9. Double DQN

Problem: standard DQN can overestimate Q-values.

Fix:

- Choose best next action using online model
- Evaluate that action using target model

Formula:

- `target = r + gamma * Q_target(s', argmax Q_model(s'))`

Purpose:

- Reduce overestimation
- Improve stability

## 10. Epsilon-Greedy

Action selection:

- If `random <= epsilon` -> random action
- Else -> `argmax` action

`epsilon` controls exploration:

- Starts high (for example `1.0`)
- Decays over time
- Never goes below `epsilon_min`

Purpose:

- Explore environment
- Avoid getting stuck in suboptimal policy

## 11. Training Control Parameters

- `batch_size`: samples per update
- `n_episodes`: total training episodes
- `gamma`: discount factor
- `learning_rate`: optimizer step size
- `update_target_every`: frequency of target network sync

## 12. CartPole Objective

- CartPole has no explicit terminal goal state to reach.
- Reward is `+1` per step survived.
- Maximum reward per episode is `500`.
- Optimal policy: consistently survive all `500` steps.
