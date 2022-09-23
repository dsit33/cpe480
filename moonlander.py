# Name:         Derek Sit
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Moonlander II
# Term:         Fall 2020

import random
from typing import Callable


class ModuleState:  # do not modify class

    def __init__(self, fuel: int, altitude: int, velocity: float, gforce: float,
                 transition: Callable[[float, float], float]) -> None:
        self.rate_max = 4
        self.fuel: int = fuel
        self.altitude: int = altitude
        self.velocity: int = velocity
        self.use_fuel: Callable[[int], ModuleState] = \
            lambda rate: ModuleState._use_fuel(fuel, altitude, velocity, gforce,
                                               transition, self.rate_max, rate)
        self.actions: Tuple[float, ...] = tuple(range(self.rate_max + 1))

    @staticmethod
    def _use_fuel(fuel: int, altitude: int, velocity: float, gforce: float,
                  transition: Callable[[float, int], float], rate_max: int,
                  rate: int) -> "ModuleState":
        fuel = max(0, fuel - rate)
        if fuel == 0:
            rate = 0
        acceleration = transition(gforce * 9.8, rate / rate_max)
        altitude = max(0, altitude + velocity + acceleration / 2)
        velocity += acceleration
        return ModuleState(fuel, altitude, velocity, gforce, transition)




def main() -> None:
    transition = lambda g, r: g * (2 * r - 1)  # example transition function
    state = ModuleState(1000, 20, 0.0, 0.1657, transition)
    pilot(state, True)


def pilot(state: ModuleState, auto: bool = True) -> None:
    if auto:
        q = learn_q(state)
        policy = lambda s: max(state.actions, key=lambda a: q(s, a))
    while state.altitude > 0:
        if auto:
            rate = policy(state)
        else:
            while True:
                rate = int(input(f"Enter Fuel Rate [0-{state.rate_max}]: "))
                if rate in state.actions:
                    break
        state = state.use_fuel(rate)
        print(f"    Fuel: {state.fuel:4} l")
        print(f"Altitude: {state.altitude:7.2f} m")
        print(f"Velocity: {state.velocity:7.2f} m/s\n")
    print(f"Impact Velocity: {state.velocity:7.2f} m/s")


def learn_q(state: ModuleState) -> Callable[[ModuleState, int], float]:
    """
    Return a Q-function that maps a state-action pair to a utility value. This
    function must be a callable with the signature shown.
    """
    dt: dict = {}
    lr: float = 0.9
    num_eps: int = 5000
    decay: float = (lr - 0.1) / num_eps
    gamma: float = 0.9
    episodes: int = 0

    dt[convert(state)] = [0 for _ in range(5)]
    while episodes < num_eps:
        current: ModuleState = state

        while current.altitude > 0:
            cur_ref: tuple = convert(current)
            action: int = best_action(cur_ref, dt)

            #modified bellman's
            dt[cur_ref][action] = (1 - lr) * dt[cur_ref][action] + lr \
                * (reward(current, state.altitude, state.fuel) + gamma * \
                    child_util(current, action, dt, state.altitude, state.fuel))

            current = current.use_fuel(action)

        #place score of landing state
        cur_ref = convert(current)
        #action = best_action(cur_ref, dt)
        if dt.get(cur_ref) == None:
            dt[cur_ref][:] = reward(current, state.altitude, state.fuel)

        lr -= decay
        episodes += 1

    return lambda s, a: dt[convert(s)][a]

def convert(current: ModuleState) -> tuple:
    return (round(current.altitude, 2), 
        round(current.velocity, 2), current.fuel)

def best_action(cur_ref: tuple, dt: dict) -> int:
    epsilon: float = 0.20

    if random.random() < epsilon:
        return random.randint(0, 4)
    best = max(dt[cur_ref])
    return dt[cur_ref].index(best)

def child_util(current: ModuleState, action: int, dt: dict, 
    start_alt: int, start_fuel: int) -> float:
    child: ModuleState = current.use_fuel(action)
    child_ref = convert(child)

    if dt.get(child_ref) == None:
        dt[child_ref] = [reward(child, start_alt, start_fuel) for _ in range(5)]
    return max(dt[child_ref])

def reward(current: ModuleState, start_alt: int, start_fuel: int) -> float:
    if current.altitude < 2:
        if current.altitude == 0:
            if current.velocity <= 0 and current.velocity >= -1:
                return 100000
            else:
                return -100000
        return -1*abs(current.velocity)

    if current.velocity > 0:
        return -1000

    return 0.7*(start_alt - current.altitude) + 0.5*current.velocity \
        - 0.2*(start_fuel - current.fuel)




if __name__ == "__main__":
    main()