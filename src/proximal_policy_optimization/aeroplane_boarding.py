import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from enum import Enum
import numpy as np 
# Register this module as a gym environment. Once registered, the id is usable in gym.make().
# When running this code, you can ignore this warning: "UserWarning: WARN: Overriding environment airplane-boarding-v0 already in registry."

register(
    id="airplane-boarding-v0",
    entry_point="airplane_boarding:AirplaneEnv",  # module_name:class_name
)
class PassengerStatus(Enum):
    MOVING  = 0
    STALLED = 1
    STOWING = 2
    SEATED  = 3

    # Returns the string representation of the PassengerStatus enum.
    def __str__(self):
        match self:
            case PassengerStatus.MOVING:
                return "MOVING"
            case PassengerStatus.STALLED:
                return "STALLED"
            case PassengerStatus.STOWING:
                return "STOWING"
            case PassengerStatus.SEATED:
                return "SEATED"
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from enum import Enum
import numpy as np

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
# When running this code, you can ignore this warning: "UserWarning: WARN: Overriding environment airplane-boarding-v0 already in registry."
register(
    id='airplane-boarding-v0',
    entry_point='airplane_boarding:AirplaneEnv', # module_name:class_name
)

class PassengerStatus(Enum):
    MOVING  = 0
    STALLED = 1
    STOWING = 2
    SEATED  = 3

    # Returns the string representation of the PassengerStatus enum.
    def __str__(self):
        match self:
            case PassengerStatus.MOVING:
                return "MOVING"
            case PassengerStatus.STALLED:
                return "STALLED"
            case PassengerStatus.STOWING:
                return "STOWING"
            case PassengerStatus.SEATED:
                return "SEATED"

class Passenger:
    def __init__(self, seat_num, row_num):
        self.seat_num = seat_num
        self.row_num = row_num
        self.is_holding_luggage = True
        self.status = PassengerStatus.MOVING

    # Returns the string representation of the Passenger class i.e. 2 digit seat number
    def __str__(self):
        return f"P{self.seat_num:02d}"

class LobbyRow:
    def __init__(self, row_num, seats_per_row):
        self.row_num = row_num
        self.passengers = [Passenger(row_num * seats_per_row + i, row_num) for i in range(seats_per_row)]

class Lobby:
    def __init__(self, num_of_rows, seats_per_row):
        self.num_of_rows = num_of_rows
        self.seats_per_row = seats_per_row
        self.lobby_rows = [LobbyRow(row_num, self.seats_per_row) for row_num in range(self.num_of_rows)]

    def remove_passenger(self, row_num):
        passenger = self.lobby_rows[row_num].passengers.pop()
        return passenger

    def count_passengers(self):
        count = 0
        for row in self.lobby_rows:
            count += len(row.passengers)

        return count

class BoardingLine:
    def __init__(self, num_of_rows):
        # Initialize the aisle
        self.num_of_rows = num_of_rows
        self.line = [None for i in range(num_of_rows)]

    def add_passenger(self, passenger):
        self.line.append(passenger)

    def is_onboarding(self):
        if (len(self.line) > 0 and not all(passenger is None for passenger in self.line)):
            return True

        return False

    def num_passengers_stalled(self):
        count = 0
        for passenger in self.line:
            if passenger is not None and passenger.status == PassengerStatus.STALLED:
                count += 1

        return count

    def num_passengers_moving(self):
        count = 0
        for passenger in self.line:
            if passenger is not None and passenger.status == PassengerStatus.MOVING:
                count += 1

        return count

    def move_forward(self):

        for i, passenger in enumerate(self.line):
            # Skip, if no passenger in that spot or
            #   passenger is at the front of the line or
            #   passenger is stowing luggage
            if passenger is None or i==0 or passenger.status == PassengerStatus.STOWING:
                continue

            # Move passenger forward, if no one is blocking
            if (passenger.status == PassengerStatus.STALLED or passenger.status == PassengerStatus.MOVING) and self.line[i-1] is None:
                passenger.status = PassengerStatus.MOVING
                self.line[i-1] = passenger
                self.line[i] = None
            else:
                passenger.status = PassengerStatus.STALLED

        # Truncate the empty spots at the end of the line
        for i in range(len(self.line)-1, self.num_of_rows-1, -1):
            if self.line[i] is None:
                self.line.pop(i)

class AirplaneEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None, num_of_rows = 3, seats_per_row = 5):

        self.seats_per_row = seats_per_row
        self.num_of_rows = num_of_rows
        self.num_of_seats = num_of_rows * seats_per_row
        self.render_mode = render_mode
        # Reset the environment
        self.reset()
        # Define the Action space
        self.action_space = spaces.Discrete(self.num_of_rows)

        # Define the Observation space.
        # The observation space is used to validate the observation returned by reset() and step().
        #[0,-1,1,-1,......6,2,7,1....]
        self.observation_space = spaces.Box(
            low=-1,
            high=self.num_of_seats-1,
            shape=(self.num_of_seats * 2,),
            dtype=np.int32
        )
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # gym requires this call to control randomness and reproduce scenarios.

        # return observation, info

    def step(self, action):
        pass

        # return observation, reward, terminated, truncated, info

    def render(self):
        pass

# Check validity of the environment
def check_env():
    from gymnasium.utils.env_checker import check_env
    env = gym.make("airplane-boarding-v0", render_mode=None)
    check_env(env.unwrapped)

if __name__ == "__main__":
    check_env()