# Name:         Derek Sit
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Mine Shafted
# Term:         Fall 2020

from typing import Generator, List, Tuple
import itertools


class BoardManager:

    def __init__(self, board: List[List[int]]):
        """
        An instance of BoardManager has two attributes.

            size: A 2-tuple containing the number of rows and columns,
                  respectively, in the game board.
            move: A callable that takes an integer as its only argument to be
                  used as the index to explore on the board. If the value at
                  that index is a clue (non-mine), this clue is returned;
                  otherwise, an error is raised.

        This constructor should only be called once per game.

        >>> board = [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]
        >>> bm = BoardManager(board)
        >>> bm.size
        (4, 3)
        >>> bm.move(4)
        2
        >>> bm.move(5)
        Traceback (most recent call last):
        ...
        RuntimeError
        """
        self.size: Tuple[int, int] = (len(board), len(board[0]))
        it: Generator[int, int, None] = BoardManager._move(board, self.size[1])
        next(it)
        self.move: Callable[[int], int] = it.send

    @staticmethod
    def _move(board: List[List[int]], width: int) -> Generator[int, int, None]:
        """
        A generator that may be sent integers (indices to explore on the board)
        and yields integers (clues for the explored indices).

        Do not call this method directly; instead, call the |move| instance
        attribute, which sends its index argument to this generator.
        """
        index = (yield 0)
        while True:
            clue = board[index // width][index % width]
            if clue == -1:
                raise RuntimeError
            index = (yield clue)

class Clue:

    def __init__(self, idx: int, domain: list):
        self.idx: int = idx
        self.domain: list = domain

    def __eq__(self, other) -> bool:
        if type(self) == type(other):
            return self.idx == other.idx
        return self.idx == other

    def __ne__(self, other) -> bool:
        if type(self) == type(other):
            return self.idx != other.idx
        return self.idx != other

    def __hash__(self) -> int:
        return hash(self.idx)

    def __str__(self) -> str:
        return str(self.idx)


def pboard(board: list, nrows: int, ncols: int) -> None:
    row = []

    for i in range(nrows):
        for j in range(ncols):
            row.append("{:>2}".format(str(board[i*ncols + j])))
        print(row)
        row.clear()

#find the adjacent unexplored spots to create relevant arcs
def adjacent_unexplored(idx: int, board: list, nrows: int, ncols: int) -> set:
    all_adj: set = set()
    adj_ue: set = set()

    all_adj = adjacent_spots(idx, board, nrows, ncols)
    
    for spot in all_adj:
        if board[spot] == -2:
            adj_ue.add(spot)

    return adj_ue

def adjacent_spots(idx: int, board: list, nrows: int, ncols: int) -> set:
    adj: set = set()
    y: int = idx // ncols
    x: int = idx % ncols

    if y > 0:
        adj.add(idx-ncols)
    if y < nrows-1:
        adj.add(idx+ncols)
    if x > 0:
        adj.add(idx-1)
    if x < ncols-1:
        adj.add(idx+1)
    if x < ncols-1 and y < nrows-1:
        adj.add(idx+ncols+1)
    if x > 0 and y < nrows-1:
        adj.add(idx+ncols-1)
    if x < ncols-1 and y > 0:
        adj.add(idx-ncols+1)
    if x > 0 and y > 0:
        adj.add(idx-ncols-1)

    return adj


def relevant_arcs(clue: Clue, clues: set, board: list, nrows: int, 
    ncols: int) -> set:

    arcs: set = set()

    adj_ue = adjacent_unexplored(clue.idx, board, nrows, ncols)
    for unexplored in adj_ue:    
        rel_clues = adjacent_spots(unexplored, board, nrows, ncols)
        rel_clues = clues.intersection(rel_clues)
        for rc in rel_clues:
            if rc != clue:
                arcs.add((clue, rc))
                arcs.add((rc, clue))

    return arcs

def absolute(num: int) -> int:
    if num < 0:
        return -1*num
    return num

def clean_clues(solved: set, clues: set, board: list, 
    nrows: int, ncols: int) -> tuple:
    
    to_delete: set = set()

    for clue in clues:
        adj = adjacent_spots(clue.idx, board, nrows, ncols)

        if adj.issubset(clues.union(solved)):
            to_delete.add(clue)

    return clues.difference(to_delete), solved.union(to_delete)


def domain(idx: int, board: list, nrows: int, ncols: int) -> list:
    all_adj: set = adjacent_spots(idx, board, nrows, ncols)
    rel_unexplored: set = set()
    dx: list = []
    discovered_bombs: int = 0

    for spot in all_adj:
        if board[spot] == -2:
            rel_unexplored.add(spot)
        if board[spot] == -1:
            discovered_bombs += 1

    if board[idx] == 0:
        dx.append(rel_unexplored)
        return dx

    nbombs = board[idx] - discovered_bombs

    #create permutations of possible bomb locations
    for comb in itertools.combinations(rel_unexplored, nbombs):
        safe = rel_unexplored.difference(comb)
        dx.append(set(map(lambda x: -1*x, comb)).union(safe))

    return dx


def reduction(dx: list, dy: list) -> tuple:
    new_dx: list = []

    for x_poss in dx:
        for y_poss in dy:

            if y_poss.issubset(x_poss) or y_poss.issuperset(x_poss):
                new_dx.append(x_poss)
                break

            num_common = len(set(map(absolute, x_poss))\
                .intersection(set(map(absolute, y_poss))))

            related = x_poss.intersection(y_poss)

            if len(related) == num_common:
                new_dx.append(x_poss)
                break

    return new_dx, (len(dx) != len(new_dx))


def propagation(clues: set, board: list, nrows: int, ncols: int) -> set:
    worklist: set = set()

    for clue in clues:
        for arc in relevant_arcs(clue, clues, board, nrows, ncols):
            worklist.add(arc)

    while len(worklist) > 0:
        #get any arc
        current: tuple = worklist.pop()
        l_clue = current[0]
        r_clue = current[1]

        if type(l_clue) == int or type(r_clue) == int:
            for c in clues:
                if l_clue == c:
                    l_clue = c
                if r_clue == c:
                    r_clue = c

        #try to reduce l_clues domain w/ respect to r_clues domain
        l_clue.domain, reduced = reduction(l_clue.domain, r_clue.domain)

        #if there was no reduction, add current to discard and move on
        if not reduced:
            continue

        #if there was, add all arcs that have l_clue as the r_clue to worklist
        clues.remove(l_clue)
        clues.add(l_clue)

        for arc in relevant_arcs(l_clue, clues, board, nrows, ncols):
            #update arcs involving reduced domain
            try:
                worklist.remove(arc)
            except KeyError:
                pass
            if arc[0] == r_clue:
                continue
            worklist.add(arc)

    return clues

def guaranteed_moves(clues: set, board: list, nrows: int, ncols: int) -> tuple:
    all_dom: set = set()
    safe: set = set()
    bombs: set = set()
    for clue in clues:
        adj_ue = adjacent_unexplored(clue.idx, board, nrows, ncols)
        for d in clue.domain:
            all_dom = all_dom.union(d)
        for unexplored in adj_ue:    
            rel_clues = adjacent_spots(unexplored, board, nrows, ncols)
            rel_clues = clues.intersection(rel_clues)
            for rc in rel_clues:
                for c in clues:
                    if rc == c:
                        rc = c
                for rcd in rc.domain:
                    all_dom = all_dom.union(rcd)

    for x in all_dom:
        if -1*x not in all_dom:
            if x < 0:
                bombs.add(x)
            if x > 0:
                safe.add(x)

    return safe, bombs

    
def make_moves(clues: set, board: list, nrows: int, ncols: int, 
    bm: BoardManager) -> tuple:

    to_add: set = set()
    to_delete: set = set()

    for clue in clues:
        if len(clue.domain) == 1:
            for spot in clue.domain[0]:
                if spot < 0:
                    board[absolute(spot)] = -1
                else:
                    board[spot] = bm.move(spot)
                    to_add.add(Clue(spot, domain(spot, board, nrows, ncols)))
            to_delete.add(clue)

    clues = clues.difference(to_delete)

    for c in clues:
        temp = domain(c.idx, board, nrows, ncols)
        if len(temp) < len(c.domain) or \
            len(temp[0]) < len(c.domain[0]):
            c.domain = temp

    clues = clues.union(to_add)

    if len(to_delete) > 0:
        return make_moves(clues, board, nrows, ncols, bm)

    return clues, board

def has_unexplored(board: list) -> bool:
    try:
        board.index(-2)
        return True
    except ValueError:
        return False

def bformat(board: list, nrows: int, ncols: int) -> list:
    new: list = []

    for row in range(nrows):
        new.append(board[row*ncols: row*ncols + ncols])

    return new


def sweep_mines(bm: BoardManager) -> List[List[int]]:
    """
    Given a BoardManager (bm) instance, return a solved board (represented as a
    2D list) by repeatedly calling bm.move(index) until all safe indices have
    been explored. If at any time a move is attempted on a non-safe index, the
    BoardManager will raise an error; this error signifies the end of the game
    and should not attempt to be caught.

    >>> board = [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]
    >>> bm = BoardManager(board)
    >>> sweep_mines(bm)
    [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]
    """
    clues: set = set()
    solved: set = set()
    nrows, ncols = bm.size
    board: list = [-2 for _ in range(nrows*ncols)]

    board[0] = bm.move(0)
    clues.add(Clue(0, domain(0, board, nrows, ncols)))

    clues, board = make_moves(clues, board, nrows, ncols, bm)
    clues, solved = clean_clues(solved, clues, board, nrows, ncols)

    #continually find next unexplored and infer bomb location
    while has_unexplored(board):
        #pboard(board, nrows, ncols)
        #print()
        clues = propagation(clues, board, nrows, ncols)
        clues, board = make_moves(clues, board, nrows, ncols, bm)
        clues, solved = clean_clues(solved, clues, board, nrows, ncols)
        safe, bombs = guaranteed_moves(clues, board, nrows, ncols)
        for s in safe:
            board[s] = bm.move(s)
            clues.add(Clue(s, domain(s, board, nrows, ncols)))
        for b in bombs:
            board[b] = -1
        clues, solved = clean_clues(solved, clues, board, nrows, ncols)


    #pboard(board, nrows, ncols)
    return bformat(board, nrows, ncols)



def main() -> None:  # optional driver
    board = [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]
    bm = BoardManager(board)
    sweep_mines(bm)
    assert sweep_mines(bm) == board

    board = [[0, 1, 1, 2, 1, 1], 
             [0, 1, -1, 2, -1, 1], 
             [0, 1, 1, 2, 1, 1],
             [1, 1, 0, 1, 1, 1],
             [-1, 1, 0, 1, -1, 1], 
             [1, 1, 0, 1, 1, 1]]
    bm = BoardManager(board)
    sweep_mines(bm)
    assert sweep_mines(bm) == board
    
    board = [[0, 0, 0, 1, -1, 1], 
             [1, 1, 1, 2, 2, 2], 
             [1, -1, 1, 1, -1, 1],
             [1, 1, 1, 2, 2, 2],
             [1, 1, 1, 1, -1, 1], 
             [1, -1, 1, 1, 1, 1]]
    bm = BoardManager(board)
    sweep_mines(bm)
    assert sweep_mines(bm) == board




if __name__ == "__main__":
    main()