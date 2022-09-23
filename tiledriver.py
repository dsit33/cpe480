# Name:         Derek Sit
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Tile Driver
# Term:         Fall 2020

import queue
from typing import List, Tuple

class State:

    def __init__(self, cost: int, moves: str, tiles: Tuple[int, ...],
     blank_pos: int):
        self.cost = cost
        self.moves = moves
        self.tiles = tiles
        self.blank_pos = blank_pos

    def f(self):
        return self.cost + len(self.moves)

    def __lt__(self, other) -> bool:
        return self.f() < other.f()

    #def __eq__(self, other) -> bool:
    #    return self.tiles == other.tiles


class Heuristic:

    @staticmethod
    def get(tiles: Tuple[int, ...]) -> int:
        """
        Return the estimated distance to the goal using Manhattan distance
        and linear conflicts.

        Only this static method should be called during a search; all other
        methods in this class should be considered private.

        >>> Heuristic.get((0, 1, 2, 3))
        0
        >>> Heuristic.get((3, 2, 1, 0))
        6
        """
        width = int(len(tiles) ** 0.5)
        return (Heuristic._get_manhattan_distance(tiles, width)
                + Heuristic._get_linear_conflicts(tiles, width))

    @staticmethod
    def _get_manhattan_distance(tiles: Tuple[int, ...], width: int) -> int:
        """
        Return the Manhattan distance of the given tiles, which represents
        how many moves is tile is away from its goal position.
        """
        distance = 0
        for i in range(len(tiles)):
            if tiles[i] != 0:
                row_dist = abs(i // width - tiles[i] // width)
                col_dist = abs(i % width - tiles[i] % width)
                distance += row_dist + col_dist
        return distance

    @staticmethod
    def _get_linear_conflicts(tiles: Tuple[int, ...], width: int) -> int:
        """
        Return the number of linear conflicts in the tiles, which represents
        the minimum number of tiles in each row and column that must leave and
        re-enter that row or column in order for the puzzle to be solved.
        """
        conflicts = 0
        rows = [[] for i in range(width)]
        cols = [[] for i in range(width)]
        for i in range(len(tiles)):
            if tiles[i] != 0:
                if i // width == tiles[i] // width:
                    rows[i // width].append(tiles[i])
                if i % width == tiles[i] % width:
                    cols[i % width].append(tiles[i])
        for i in range(width):
            conflicts += Heuristic._count_conflicts(rows[i])
            conflicts += Heuristic._count_conflicts(cols[i])
        return conflicts * 2

    @staticmethod
    def _count_conflicts(ints: List[int]) -> int:
        """
        Return the minimum number of tiles that must be removed from the given
        list in order for the list to be sorted.
        """
        if Heuristic._is_sorted(ints):
            return 0
        lowest = None
        for i in range(len(ints)):
            conflicts = Heuristic._count_conflicts(ints[:i] + ints[i + 1:])
            if lowest is None or conflicts < lowest:
                lowest = conflicts
        return 1 + lowest

    @staticmethod
    def _is_sorted(ints: List[int]) -> bool:
        """Return True if the given list is sorted and False otherwise."""
        for i in range(len(ints) - 1):
            if ints[i] > ints[i + 1]:
                return False
        return True


def find_blank(tiles: Tuple[int, ...]) -> int:
    if 0 in tiles:
        return tiles.index(0)
    return -1

def solve_puzzle(tiles: Tuple[int, ...]) -> str:
    """
    Return a string (containing characters "H", "J", "K", "L") representing the
    optimal number of moves to solve the given puzzle.
    """
    frontier = queue.PriorityQueue()
    blank_pos = find_blank(tiles)
    start = State(Heuristic.get(tiles), "", tiles, blank_pos)
    frontier.put(start)
    width = int(len(tiles) ** 0.5)
    
    #get the most promising state from the frontier and generate potential
    #states until a solution is found

    while not frontier.empty():
        curr = frontier.get()

        #we reached solution
        if curr.cost == 0:
            return curr.moves
        
        for s in gen_states(curr, width):
            frontier.put(s)

    return "no solution"


#the following 4 functions will generate potential moves
def gen_right(tiles: Tuple[int, ...], moves: str, blank_pos: int, 
    width: int) -> State:
    new = list(tiles)

    new[blank_pos] = new[blank_pos - 1]
    new[blank_pos - 1] = 0

    tup = tuple(new)

    return State(Heuristic.get(tup), moves + 'L', tup, blank_pos-1)

def gen_left(tiles: Tuple[int, ...], moves: str, blank_pos: int, 
    width: int) -> State:
    new = list(tiles)

    new[blank_pos] = new[blank_pos + 1]
    new[blank_pos + 1] = 0

    tup = tuple(new)

    return State(Heuristic.get(tup), moves + 'H', tup, blank_pos+1)

def gen_up(tiles: Tuple[int, ...], moves: str, blank_pos: int, 
    width: int) -> State:
    new = list(tiles)

    new[blank_pos] = new[blank_pos + width]
    new[blank_pos + width] = 0

    tup = tuple(new)

    return State(Heuristic.get(tup), moves + 'K', tup, blank_pos+width)

def gen_down(tiles: Tuple[int, ...], moves: str, blank_pos: int, 
    width: int) -> State:
    new = list(tiles)

    new[blank_pos] = new[blank_pos - width]
    new[blank_pos - width] = 0

    tup = tuple(new)

    return State(Heuristic.get(tup), moves + 'J', tup, blank_pos-width)


#generate a list of the new potential states based on curr
def gen_states(curr: State, width: int) -> List[State]:

    blank_pos = curr.blank_pos
    states = []
    last_move = ""

    #curr.moves only empty if this is the initial state
    if curr.moves != "":
        last_move = curr.moves[-1]

    col = blank_pos % width
    row = blank_pos // width

    if col != width - 1 and last_move != 'L':
        states.append(gen_left(curr.tiles, curr.moves, blank_pos, width))

    if row != 0 and last_move != 'K':
        states.append(gen_down(curr.tiles, curr.moves, blank_pos, width))

    if row != width - 1 and last_move != 'J':
        states.append(gen_up(curr.tiles, curr.moves, blank_pos, width))

    if col != 0 and last_move != 'H':
        states.append(gen_right(curr.tiles, curr.moves, blank_pos, width))

    return states
    


def main() -> None:
    """Optional: Use as a driver to test your program."""
    tiles = (6, 7, 8, 3, 0, 5, 1, 2, 4)
    print(solve_puzzle(tiles))

    



if __name__ == "__main__":
    main()
