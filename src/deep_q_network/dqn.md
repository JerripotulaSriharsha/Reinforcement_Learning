# What Is Common in CartPole and Other DQN Problems?

## When DQN Works

### 1. State is continuous (or large)

Examples:

- CartPole -> 4 floats
- Atari -> image pixels
- Game environments -> complex observations

DQN uses neural networks to approximate `Q(s, a)`.

### 2. Actions are discrete

This is essential.

CartPole:

- `left (0)`
- `right (1)`

Atari:

- `up, down, left, right, fire, ...`

Still discrete.

DQN requires a finite number of discrete actions.

### 3. Goal is to maximize cumulative reward

CartPole:

- Reward = `+1` per step
- Maximize survival time

Atari:

- Reward = game score
- Maximize score

GridWorld:

- Reward at goal
- Maximize total reward

Same objective across environments.

## When DQN Does Not Work

DQN is not suitable when actions are continuous.

Example:

- Robot arm torque: `action = 0.347` Newton-meters
- Infinite action possibilities

You cannot efficiently do `argmax` over infinite actions.

Use actor-critic methods instead:

- DDPG
- SAC
- TD3

## What Changes Between Environments?

| Component | CartPole | Atari | Other |
|---|---|---|---|
| State size | 4 | 84x84x4 image | Depends |
| Action size | 2 | 6+ | Depends |
| Reward design | +1 per step | Game score | Depends |
| Termination | Pole falls | Lose life/game over | Depends |

Even when these change, the DQN training loop stays the same.

## What Is Universal in DQN?

Always:

1. `state -> Q-network -> Q-values`
2. `argmax -> action`
3. `environment -> (next_state, reward, done)`
4. compute target
5. backpropagation update
6. repeat

This structure is consistent across DQN tasks.

## What Is Special About CartPole?

Nothing algorithmically special.

CartPole is simply:

- Simple
- Low-dimensional
- Easy to visualize
- Fast to train

It is a teaching benchmark.

## Real-World Example (Discrete Actions)

Self-driving lane decision:

- State: camera features + speed
- Actions: left lane, stay, right lane
- Reward: safety + efficiency

This still fits DQN because actions are discrete.

## Final Summary

DQN works best for:

- Large or continuous state spaces
- Discrete action spaces
- Reward maximization objectives

CartPole is the simplest demo of that setup.
