
"""Single-file airplane boarding environment with optional emoji-based rendering.

This version keeps the RL environment logic in one file and adds a simple
matplotlib renderer so the simulation can be visualised in a style similar to
the screenshot the user shared.
"""

"""Single-file airplane boarding environment with optional emoji-based rendering.

This version keeps the RL environment logic in one file and adds a simple
matplotlib renderer so the simulation can be visualised in a style similar to
the screenshot the user shared.
"""

from __future__ import annotations

import os
import time
from enum import Enum
from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register


ENV_ID = "airplane-boarding-emoji-v0"
MODEL_DIR = "models"
LOG_DIR = "logs"


try:
    register(
        id=ENV_ID,
        entry_point="airplane_boarding_emoji_single_file:AirplaneEnv",
    )
except Exception:
    pass


class PassengerStatus(Enum):
    MOVING = 0
    STALLED = 1
    STOWING = 2
    SEATED = 3


class Passenger:
    def __init__(self, seat_num: int, row_num: int) -> None:
        self.seat_num = seat_num
        self.row_num = row_num
        self.is_holding_luggage = True
        self.status = PassengerStatus.MOVING

    def __str__(self) -> str:
        return f"P{self.seat_num:02d}"


class LobbyRow:
    def __init__(self, row_num: int, seats_per_row: int) -> None:
        self.row_num = row_num
        self.passengers = [
            Passenger(row_num * seats_per_row + i, row_num)
            for i in range(seats_per_row)
        ]


class Lobby:
    def __init__(self, num_of_rows: int, seats_per_row: int) -> None:
        self.lobby_rows = [
            LobbyRow(row_num, seats_per_row) for row_num in range(num_of_rows)
        ]

    def remove_passenger(self, row_num: int) -> Passenger:
        return self.lobby_rows[row_num].passengers.pop()

    def count_passengers(self) -> int:
        return sum(len(row.passengers) for row in self.lobby_rows)


class BoardingLine:
    def __init__(self, num_of_rows: int) -> None:
        self.num_of_rows = num_of_rows
        self.line: list[Optional[Passenger]] = [None for _ in range(num_of_rows)]

    def add_passenger(self, passenger: Passenger) -> None:
        self.line.append(passenger)

    def is_onboarding(self) -> bool:
        return len(self.line) > 0 and not all(p is None for p in self.line)

    def num_passengers_stalled(self) -> int:
        return sum(
            1
            for passenger in self.line
            if passenger is not None and passenger.status == PassengerStatus.STALLED
        )

    def move_forward(self) -> None:
        for i, passenger in enumerate(self.line):
            if passenger is None or i == 0 or passenger.status == PassengerStatus.STOWING:
                continue

            if self.line[i - 1] is None:
                passenger.status = PassengerStatus.MOVING
                self.line[i - 1] = passenger
                self.line[i] = None
            else:
                passenger.status = PassengerStatus.STALLED

        for i in range(len(self.line) - 1, self.num_of_rows - 1, -1):
            if self.line[i] is None:
                self.line.pop(i)


class Seat:
    def __init__(self, seat_num: int, row_num: int) -> None:
        self.seat_num = seat_num
        self.row_num = row_num
        self.passenger: Optional[Passenger] = None

    def seat_passenger(self, passenger: Passenger) -> bool:
        if self.seat_num != passenger.seat_num:
            raise AssertionError("Seat number does not match passenger seat number.")

        if passenger.is_holding_luggage:
            passenger.status = PassengerStatus.STOWING
            passenger.is_holding_luggage = False
            return False

        self.passenger = passenger
        self.passenger.status = PassengerStatus.SEATED
        return True


class AirplaneRow:
    def __init__(self, row_num: int, seats_per_row: int) -> None:
        self.row_num = row_num
        self.seats = [
            Seat(row_num * seats_per_row + i, row_num) for i in range(seats_per_row)
        ]

    def try_sit_passenger(self, passenger: Passenger) -> bool:
        for seat in self.seats:
            if seat.seat_num == passenger.seat_num:
                return seat.seat_passenger(passenger)
        return False


class AirplaneEnv(gym.Env):
    """Airplane boarding environment with terminal and emoji rendering."""

    metadata = {"render_modes": ["human", "terminal", "emoji"], "render_fps": 2}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        num_of_rows: int = 10,
        seats_per_row: int = 5,
        render_pause: float = 0.35,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.num_of_rows = num_of_rows
        self.seats_per_row = seats_per_row
        self.num_of_seats = num_of_rows * seats_per_row
        self.render_pause = render_pause

        self.action_space = spaces.Discrete(self.num_of_rows)
        self.observation_space = spaces.Box(
            low=-1,
            high=self.num_of_seats - 1,
            shape=(self.num_of_seats * 2,),
            dtype=np.int32,
        )

        self.fig = None
        self.ax = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.airplane_rows = [
            AirplaneRow(row_num, self.seats_per_row)
            for row_num in range(self.num_of_rows)
        ]
        self.lobby = Lobby(self.num_of_rows, self.seats_per_row)
        self.boarding_line = BoardingLine(self.num_of_rows)
        self.render()
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        observation: list[int] = []
        for passenger in self.boarding_line.line:
            if passenger is None:
                observation.extend([-1, -1])
            else:
                observation.extend([passenger.seat_num, passenger.status.value])

        for _ in range(len(self.boarding_line.line), self.num_of_seats):
            observation.extend([-1, -1])

        return np.array(observation, dtype=np.int32)

    def step(self, row_num: int):
        if not (0 <= row_num < self.num_of_rows):
            raise AssertionError("Action row number is out of range.")

        reward = 0

        if self.lobby.count_passengers() > 0:
            passenger = self.lobby.remove_passenger(row_num)
            self.boarding_line.add_passenger(passenger)
            self._move()
            reward = self._calculate_reward()
        else:
            while self.is_onboarding():
                self._move()
                reward += self._calculate_reward()

        terminated = not self.is_onboarding()
        return self._get_observation(), reward, terminated, False, {}

    def _calculate_reward(self) -> int:
        return -self.boarding_line.num_passengers_stalled()

    def is_onboarding(self) -> bool:
        return self.lobby.count_passengers() > 0 or self.boarding_line.is_onboarding()

    def _move(self) -> None:
        for row_num, passenger in enumerate(self.boarding_line.line):
            if passenger is None:
                continue
            if row_num >= len(self.airplane_rows):
                break
            if self.airplane_rows[row_num].try_sit_passenger(passenger):
                self.boarding_line.line[row_num] = None

        self.boarding_line.move_forward()
        self.render()

    def render(self) -> None:
        if self.render_mode == "terminal":
            self._render_terminal()
        elif self.render_mode in {"emoji", "human"}:
            self._render_emoji()

    def _render_terminal(self) -> None:
        print("Seats".center(19) + " | Aisle Line")
        for row in self.airplane_rows:
            for seat in row.seats:
                label = f"P{seat.passenger.seat_num:02d}" if seat.passenger else f"S{seat.seat_num:02d}"
                print(label, end=" ")

            if row.row_num < len(self.boarding_line.line):
                passenger = self.boarding_line.line[row.row_num]
                status = "" if passenger is None else passenger.status.name
                print(f"| {passenger} {status}", end=" ")
            print()

        print("\nLine entering plane:")
        for i in range(self.num_of_rows, len(self.boarding_line.line)):
            passenger = self.boarding_line.line[i]
            if passenger is not None:
                print(f"{passenger} {passenger.status.name}")

        print("\nLobby:")
        for row in self.lobby.lobby_rows:
            if row.passengers:
                print(" ".join(str(passenger) for passenger in row.passengers))
        print()

    def _setup_figure(self) -> None:
        if self.fig is None or self.ax is None:
            plt.ion()
            fig_height = max(7, self.num_of_rows * 0.9)
            self.fig, self.ax = plt.subplots(figsize=(10, fig_height))

    def _draw_seat(self, x: float, y: float, occupied: bool, seat_num: int) -> None:
        seat_box = patches.FancyBboxPatch(
            (x - 0.38, y - 0.32),
            0.76,
            0.64,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#4f81bd",
            facecolor="#7ec8ff" if occupied else "#ffffff",
        )
        self.ax.add_patch(seat_box)
        if occupied:
            self.ax.text(x, y + 0.02, "🙂", ha="center", va="center", fontsize=28)
            self.ax.text(x, y + 0.35, f"{seat_num:02d}", ha="center", va="center", fontsize=10)
        else:
            self.ax.text(x, y + 0.35, f"{seat_num:02d}", ha="center", va="center", fontsize=9, alpha=0.5)

    def _draw_aisle_passenger(self, x: float, y: float, passenger: Passenger) -> None:
        face = "😐" if passenger.status == PassengerStatus.STOWING else "🙂"
        luggage = "💼" if passenger.is_holding_luggage else ""
        self.ax.text(x, y + 0.02, face, ha="center", va="center", fontsize=28)
        self.ax.text(x + 0.18, y - 0.14, luggage, ha="center", va="center", fontsize=18)
        self.ax.text(x, y + 0.36, f"{passenger.seat_num:02d}", ha="center", va="center", fontsize=10)

    def _render_emoji(self) -> None:
        self._setup_figure()
        self.ax.clear()

        left_cols = 2
        right_cols = self.seats_per_row - left_cols
        aisle_x = 2.5
        left_xs = [0.0, 1.0]
        right_xs = [3.5 + i for i in range(right_cols)]

        for row_idx, row in enumerate(self.airplane_rows):
            y = self.num_of_rows - 1 - row_idx

            for col_idx, seat in enumerate(row.seats[:left_cols]):
                x = left_xs[col_idx]
                self._draw_seat(x, y, seat.passenger is not None, seat.seat_num)

            aisle_box = patches.FancyBboxPatch(
                (aisle_x - 0.42, y - 0.42),
                0.84,
                0.84,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=1.0,
                edgecolor="#dddddd",
                facecolor="#fafafa",
            )
            self.ax.add_patch(aisle_box)

            if row_idx < len(self.boarding_line.line):
                aisle_passenger = self.boarding_line.line[row_idx]
                if aisle_passenger is not None:
                    self._draw_aisle_passenger(aisle_x, y, aisle_passenger)

            for col_idx, seat in enumerate(row.seats[left_cols:]):
                x = right_xs[col_idx]
                self._draw_seat(x, y, seat.passenger is not None, seat.seat_num)

        queue_y = -0.9
        queue_start_x = aisle_x
        for i in range(self.num_of_rows, len(self.boarding_line.line)):
            passenger = self.boarding_line.line[i]
            if passenger is not None:
                x = queue_start_x + (i - self.num_of_rows) * 0.85
                self._draw_aisle_passenger(x, queue_y, passenger)

        self.ax.set_xlim(-0.8, max(right_xs) + 1.0)
        self.ax.set_ylim(-1.5, self.num_of_rows - 0.2)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.ax.set_title("Airplane Boarding Simulation", fontsize=14)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(self.render_pause)

    def close(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    def action_masks(self) -> list[bool]:
        return [len(row.passengers) > 0 for row in self.lobby.lobby_rows]


def train(
    total_timesteps: int = int(1e6),
    num_of_rows: int = 10,
    seats_per_row: int = 5,
    n_envs: int = 12,
) -> None:
    if MaskablePPO is None:
        raise ImportError(
            "Training dependencies are missing. Install sb3-contrib and stable-baselines3."
        )

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    env = make_vec_env(
        AirplaneEnv,
        n_envs=n_envs,
        env_kwargs={"num_of_rows": num_of_rows, "seats_per_row": seats_per_row},
        vec_env_cls=SubprocVecEnv,
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        tensorboard_log=LOG_DIR,
        ent_coef=0.05,
    )

    eval_callback = MaskableEvalCallback(
        env,
        eval_freq=10_000,
        verbose=1,
        best_model_save_path=os.path.join(MODEL_DIR, "MaskablePPO"),
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)


def test(
    model_name: str,
    num_of_rows: int = 10,
    seats_per_row: int = 5,
    render: bool = True,
) -> None:
    if MaskablePPO is None or get_action_masks is None:
        raise ImportError(
            "Testing dependencies are missing. Install sb3-contrib and stable-baselines3."
        )

    env = gym.make(
        ENV_ID,
        num_of_rows=num_of_rows,
        seats_per_row=seats_per_row,
        render_mode="emoji" if render else None,
    )

    model = MaskablePPO.load(f"{MODEL_DIR}/MaskablePPO/{model_name}", env=env)

    total_reward = 0
    obs, _ = env.reset()
    terminated = False

    while not terminated:
        action_masks = get_action_masks(env)
        action, _ = model.predict(
            observation=obs,
            deterministic=True,
            action_masks=action_masks,
        )
        obs, reward, terminated, _, _ = env.step(action)
        total_reward += reward

    print(f"Total reward: {total_reward}")
    env.close()


def demo_random(num_of_rows: int = 10, seats_per_row: int = 5, pause: float = 0.35) -> None:
    """Run a random valid-policy demo with the emoji renderer."""
    env = AirplaneEnv(
        num_of_rows=num_of_rows,
        seats_per_row=seats_per_row,
        render_mode="emoji",
        render_pause=pause,
    )
    _, _ = env.reset()
    terminated = False

    while not terminated:
        valid_actions = [i for i, ok in enumerate(env.action_masks()) if ok]
        action = int(np.random.choice(valid_actions))
        _, _, terminated, _, _ = env.step(action)

    time.sleep(1.5)
    env.close()


demo_random()
