import argparse
import os

import gymnasium as gym
import numpy as np
import pygame
from keras import Model, layers
from keras.models import load_model


class DQN(Model):
    def __init__(self, action_size, **kwargs):
        super(DQN, self).__init__(**kwargs)
        self.action_size = action_size
        self.d1 = layers.Dense(24, activation="relu", name="d1")
        self.d2 = layers.Dense(24, activation="relu", name="d2")
        self.d3 = layers.Dense(action_size, activation="linear", name="d3")

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)

    def get_config(self):
        config = super(DQN, self).get_config()
        config.update({"action_size": self.action_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def run(model_path: str, episodes: int = 2, fps: int = 60) -> None:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load_model(model_path, custom_objects={"DQN": DQN}, compile=False)

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    pygame.init()

    try:
        for episode in range(episodes):
            state, _ = env.reset()
            state = state.reshape(1, -1)
            total_reward = 0.0
            done = False

            frame = env.render()
            height, width, _ = frame.shape
            screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("CartPole DQN (Pygame)")
            clock = pygame.time.Clock()

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True

                q_values = model.predict(state, verbose=0)
                action = int(np.argmax(q_values[0]))

                next_state, reward, terminated, truncated, _ = env.step(action)
                state = next_state.reshape(1, -1)
                total_reward += reward
                done = done or terminated or truncated

                frame = env.render()
                surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                screen.blit(surface, (0, 0))
                pygame.display.flip()
                clock.tick(fps)

            print(f"Episode {episode + 1}/{episodes} reward: {total_reward}")
    finally:
        env.close()
        pygame.quit()


def parse_args() -> argparse.Namespace:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.join(base_dir, "cartpole_model", "model_500.keras")

    parser = argparse.ArgumentParser(description="Render CartPole with a trained DQN model in Pygame")
    parser.add_argument("--model", default=default_model, help="Path to .keras model file")
    parser.add_argument("--episodes", type=int, default=2, help="Number of episodes to render")
    parser.add_argument("--fps", type=int, default=60, help="Display FPS cap")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(model_path=args.model, episodes=args.episodes, fps=args.fps)
