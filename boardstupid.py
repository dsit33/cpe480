# Name:         Derek Sit
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Board Stupid
# Term:         Fall 2020

import math
import random
from typing import Callable, Generator, Optional, Tuple


class GameState:

    def __init__(self, board: Tuple[Tuple[Optional[int]]], player: int) -> None:
        """
        An instance of GameState has the following attributes.

            player: Set as either 1 (MAX) or -1 (MIN).
            moves: A tuple of integers representing empty indices of the board.
            selected: The index that the current player believes to be their
                      optimal move; defaults to -1.
            util: The utility of the board; either 1 (MAX wins), -1 (MIN wins),
                  0 (tie game), or None (non-terminal game state).
            traverse: A callable that takes an integer as its only argument to
                      be used as the index to apply a move on the board,
                      returning a new GameState object with this move applied.
                      This callable provides a means to traverse the game tree
                      without modifying parent states.
            display: A string representation of the board, which should only be
                     used for debugging and not parsed for strategy.

        In addition, instances of GameState may be stored in hashed
        collections, such as sets or dictionaries.
        >>> board = ((   0,    0,    0,  \
                         0,    0, None,  \
                         0, None,    0), \
                     (   0,    0, None, \
                         0,    0, None, \
                      None, None, None), \
                     (   0, None,    0, \
                      None, None, None, \
                         0, None,    0))
        >>> state = GameState(board, 1)
        >>> state.util
        None
        >>> state.player
        1
        >>> state.moves
        (0, 1, 2, 3, 4, 6, 8, 9, 10, 12, 13, 18, 20, 24, 26)
        >>> state = state.traverse(2)
        >>> state.player
        -1
        >>> state.moves
        (0, 1, 3, 4, 6, 9, 10, 12, 13, 18, 20, 24, 26)
        >>> state = state.traverse(8)
        >>> state.player
        1
        >>> state.moves
        (0, 3, 4, 6, 9, 10, 12, 13, 18, 20, 24, 26)
        >>> state = state.traverse(1)
        >>> state.player
        -1
        >>> state.moves
        (0, 3, 6, 9, 10, 12, 13, 18, 20, 24, 26)
        >>> state = state.traverse(4)
        >>> state.player
        1
        >>> state.moves
        (3, 6, 9, 10, 12, 13, 18, 20, 24, 26)
        >>> state = state.traverse(0)
        >>> state.util
        1
        """
        self.player: int = player
        self.moves: Tuple[int] = GameState._get_moves(board, len(board))
        self.selected: int = -1
        self.util: Optional[int] = GameState._get_utility(board, len(board))
        self.traverse: Callable[[int], GameState] = \
            lambda index: GameState._traverse(board, len(board), player, index)
        self.display: str = GameState._to_string(board, len(board))
        self.keys: Tuple[int, ...] = tuple(hash(single) for single in board)

    def __eq__(self, other: "GameState") -> bool:
        return self.keys == other.keys

    def __hash__(self) -> int:
        return hash(self.keys)

    @staticmethod
    def _traverse(board: Tuple[Tuple[Optional[int]]],
                  width: int, player: int, index: int) -> "GameState":
        """
        Return a GameState instance in which the board is updated at the given
        index by the current player.

        Do not call this method directly; instead, call the |traverse| instance
        attribute, which only requires an index as an argument.
        """
        i, j = index // width ** 2, index % width ** 2
        single = board[i][:j] + (player,) + board[i][j + 1:]
        return GameState(board[:i] + (single,) + board[i + 1:], -player)

    @staticmethod
    def _get_moves(board: Tuple[Tuple[Optional[int]]],
                   width: int) -> Tuple[int]:
        """
        Return a tuple of the unoccupied indices remaining on the board.
        """
        return tuple(j + i * width ** 2 for i, single in enumerate(board)
                     for j, square in enumerate(single) if square == 0)

    @staticmethod
    def _get_utility(board: Tuple[Tuple[Optional[int]]],
                     width: int) -> Optional[int]:
        """
        Return the utility of the board; either 1 (MAX wins), -1 (MIN wins),
        0 (tie game), or None (non-terminal game state).
        """
        for line in GameState._iter_lines(board, width):
            if line == (1,) * width:
                return 1
            if line == (-1,) * width:
                return -1
        return 0 if len(GameState._get_moves(board, width)) == 0 else None

    @staticmethod
    def _iter_lines(board: Tuple[Tuple[Optional[int]]],
                    width: int) -> Generator[Tuple[int], None, None]:
        """
        Iterate over all groups of indices that represent a winning condition.
        X lines are row-wise, Y lines are column-wise, and Z lines go through
        all single boards; combinations of these axes refer to the direction
        of the line in 2D or 3D space.
        """
        for single in board:
            # x lines (2D rows)
            for i in range(0, len(single), width):
                yield single[i:i + width]
            # y lines (2D columns)
            for i in range(width):
                yield single[i::width]
            # xy lines (2D diagonals)
            yield single[::width + 1]
            yield single[width - 1:len(single) - 1:width - 1]
        # z lines
        for i in range(width ** 2):
            yield tuple(single[i] for single in board)
        for j in range(width):
            # xz lines
            yield tuple(board[i][j * width + i] for i in range(len(board)))
            yield tuple(board[i][j * width + width - 1 - i]
                        for i in range(len(board)))
            # yz lines
            yield tuple(board[i][j + i * width] for i in range(len(board)))
            yield tuple(board[i][-j - 1 - i * width]
                        for i in range(len(board)))
        # xyz lines
        yield tuple(board[i][i * width + i] for i in range(len(board)))
        yield tuple(board[i][i * width + width - 1 - i]
                    for i in range(len(board)))
        yield tuple(board[i][width ** 2 - width * (i + 1) + i]
                    for i in range(len(board)))
        yield tuple(board[i][width ** 2 - (i * width) - i - 1]
                    for i in range(len(board)))

    @staticmethod
    def _to_string(board: Tuple[Tuple[Optional[int]]], width: int) -> str:
        """
        Return a string representation of the game board, in which integers
        represent the indices of empty spaces and the characters "X" and "O"
        represent previous move selections for MAX and MIN, repsectively.
        """
        display = "\n"
        for i in range(width):
            for j in range(width):
                line = board[j][i * width:i * width + width]
                start = j * width ** 2 + i * width
                for k, space in enumerate(line):
                    if space == 0:
                        space = start + k
                    else:
                        space = "X" if space == 1 else "O"
                    display += "{0:>4}".format(space)
                display += " " * width
            display += "\n"
        return display

class Node:

    def __init__(self, state: GameState, move: int):
        self.state: GameState = state
        self.move: int = move
        self.children: dict = {}
        self.plays: int = 0
        self.wins: float = 0.0
        self.losses: float = 0.0
        self.ucb: float = 1.4
        self.frontier: PQ = PQ()

    def __eq__(self, other):
        return hash(self.state.board) == hash(other.state.board)

    def __hash__(self):
        return hash(self.state)

    def wr(self) -> float:
        if self.plays > 0:
            if self.state.player == 1:
                return float(self.wins) / float(self.plays)
            return float(self.losses) / float(self.plays)
        return 0

def by_ucb(node: Node) -> float:
    return node.ucb

class PQ:

    def __init__(self):
        self.items: list[Node] = []

    def empty(self) -> bool:
        return len(self.items) == 0

    def push(self, item: Node):
        self.items.append(item)
        self.items.sort(key=by_ucb, reverse=True)

    def pop(self, player: int) -> Node:
        if not self.empty():
            if player == -1:
                temp = self.items[0]
                self.items = self.items[1:]
            else:
                temp = self.items[-1]
                self.items = self.items[:-1]
            return temp
        else:
            return None

    def __str__(self):
        for item in self.items:
            print(item.ucb)


def main() -> None:
    board = tuple(tuple(0 for _ in range(i, i + 16))
                  for i in range(0, 64, 16))
    state = GameState(board, 1)
    while state.util is None:
        # human move
        print(state.display)
        state = state.traverse(int(input("Move: ")))
        if state.util is not None:
            break
        # computer move
        find_best_move(state)
        move = (state.selected if state.selected != -1
                else random.choice(state.moves))
        state = state.traverse(move)
    print(state.display)
    if state.util == 0:
        print("Tie Game")
    else:
        print(f"Player {state.util} Wins!")


def find_best_move(state: GameState) -> None:
    """
    Search the game tree for the optimal move for the current player, storing
    the index of the move in the given GameState object's selected attribute.
    The move must be an integer indicating an index in the 3D board - ranging
    from 0 to 63 - with 0 as the index of the top-left space of the top board
    and 63 as the index of the bottom-right space of the bottom board.

    This function must perform a Monte Carlo Tree Search to select a move,
    using additional functions as necessary. During the search, whenever a
    better move is found, the selected attribute should be immediately updated
    for retrieval by the instructor's game driver. Each call to this function
    will be given a set number of seconds to run; when the time limit is
    reached, the index stored in selected will be used for the player's turn.
    """

    best: Node = Node(state, -1)
    root: Node = Node(state, -1)
    best.wins = -1
    best.plays = 1

    while True:
        #generate the frontier of states to be considered
        root.frontier = fill_frontier(root)

        #MCTS
        while not root.frontier.empty():
            current: Node = root.frontier.pop(-1)
            #print("move:", current.move)

            #winning move
            if current.state.util == -1:
                state.selected = current.move
                break

            #explore this node and update tree
            sample_rollouts(root, current, 20)

            if current.wr() >= best.wr():
                best = current
                state.selected = best.move


            root.children[current.move] = current
            root.frontier.push(current)


def should_explore(node: Node, player: int) -> bool:
    return node.plays > 25 and node.wr() < 0.60 #and player == -1) \
        #or (node.wr() > 0.50 and player == 1))

def rollout(current: Node) -> tuple:
    temp: GameState = current.state
    move: int = -1

    #winning state
    if temp.util != None:
        current = analyze(temp.util, current)
        return current, temp.util

    #get an already explored child
    if not current.frontier.empty():
        best_child = current.frontier.pop(temp.player)

        #get new random child if current best is bad
        if should_explore(best_child, temp.player):
            remaining = set(temp.moves).difference(
                set(current.children.keys()))
            move = random.choice(list(remaining))
            temp = temp.traverse(move)
            best_child = Node(temp, move)

        best_child, result = rollout(best_child)
        best_child = analyze(result, best_child)
        current = analyze(result, current)

        best_child.ucb = best_child.wr() + 1.4 * math.sqrt(
            math.log(current.plays)/best_child.plays)

        current.frontier.push(best_child)
        current.children[move] = best_child

        return current, result
    #random playout for new path
    else:
        move = random.choice(temp.moves)
        temp = temp.traverse(move)
        child = Node(temp, move)
        
        while temp.util == None:
            move = random.choice(temp.moves)
            temp = temp.traverse(move)

        current = analyze(temp.util, current)
        child = analyze(temp.util, child)

        current.children[move] = child
        current.frontier.push(child)

        return current, temp.util
    

def analyze(result: int, current: Node) -> Node:
    if result == 1:
        current.losses += 1
    elif result == 0:
        current.wins += 0.5
        current.losses += 0.5
    elif result == -1:
        current.wins += 1

    current.plays += 1

    return current


def sample_rollouts(root: Node, node: Node, n: int) -> None:
    for _ in range(n):
        node, result = rollout(node)

        root = analyze(result, root)

        node.ucb = node.wr() + 1.4 * math.sqrt(
            math.log(root.plays)/node.plays)

def fill_frontier(root: Node) -> PQ:
    frontier: PQ = PQ()

    for move in root.state.moves:
        child: Node = Node(root.state.traverse(move), move)
        sample_rollouts(root, child, 10)
        frontier.push(child)
        root.children[move] = child

    return frontier



if __name__ == "__main__":
    main()