# Name:         Derek Sit
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Biogimmickry
# Term:         Fall 2020

import random
from typing import Callable, Dict, Tuple

class Indv:

    def __init__(self, prog: str, score: int) -> None:
        self.prog = prog
        self.score = score
        self.prob = 0

class FitnessEvaluator:

    def __init__(self, array: Tuple[int, ...]) -> None:
        """
        An instance of FitnessEvaluator has one attribute, which is a callable.

            evaluate: A callable that takes a program string as its only
                      argument and returns an integer indicating how closely
                      the program populated the target array, with a return
                      value of zero meaning the program was accurate.

        This constructor should only be called once per search.

        >>> fe = FitnessEvaluator((0, 20))
        >>> fe.evaulate(">+")
        19
        >>> fe.evaulate("+++++[>++++<-]")
        0
        """
        self.evaluate: Callable[[str], int] = \
            lambda program: FitnessEvaluator._evaluate(array, program)

    @staticmethod
    def interpret(program: str, size: int) -> Tuple[int, ...]:
        """
        Using a zeroed-out memory array of the given size, run the given
        program to update the integers in the array. If the program is
        ill-formatted or requires too many iterations to interpret, raise a
        RuntimeError.
        """
        p_ptr = 0
        a_ptr = 0
        count = 0
        max_steps = 1000
        i_map = FitnessEvaluator._preprocess(program)
        memory = [0] * size
        while p_ptr < len(program):
            if program[p_ptr] == "[":
                if memory[a_ptr] == 0:
                    p_ptr = i_map[p_ptr]
            elif program[p_ptr] == "]":
                if memory[a_ptr] != 0:
                    p_ptr = i_map[p_ptr]
            elif program[p_ptr] == "<":
                if a_ptr > 0:
                    a_ptr -= 1
            elif program[p_ptr] == ">":
                if a_ptr < len(memory) - 1:
                    a_ptr += 1
            elif program[p_ptr] == "+":
                memory[a_ptr] += 1
            elif program[p_ptr] == "-":
                memory[a_ptr] -= 1
            else:
                raise RuntimeError
            p_ptr += 1
            count += 1
            if count > max_steps:
                raise RuntimeError
        return tuple(memory)

    @staticmethod
    def _preprocess(program: str) -> Dict[int, int]:
        """
        Return a dictionary mapping the index of each [ command with its
        corresponding ] command. If the program is ill-formatted, raise a
        RuntimeError.
        """
        i_map = {}
        stack = []
        for p_ptr in range(len(program)):
            if program[p_ptr] == "[":
                stack.append(p_ptr)
            elif program[p_ptr] == "]":
                if len(stack) == 0:
                    raise RuntimeError
                i = stack.pop()
                i_map[i] = p_ptr
                i_map[p_ptr] = i
        if len(stack) != 0:
            raise RuntimeError
        return i_map

    @staticmethod
    def _evaluate(expect: Tuple[int, ...], program: str) -> int:
        """
        Return the sum of absolute differences between each index in the given
        tuple and the memory array created by interpreting the given program.
        """
        actual = FitnessEvaluator.interpret(program, len(expect))
        return sum(abs(x - y) for x, y in zip(expect, actual))


def init_population(max_len: int, pop_size: int) -> list:
    pop: set = set()

    while len(pop) < pop_size:
        prog = rand_prog(max_len)
        pop.add(prog)

    return list(pop)
        
def rand_prog(max_len: int) -> str:
    rand = low = high = count = 0
    prog = ""
    add_com = ''

    if max_len == 0:
        prog_len = random.randint(3, 150)
        l_prob = .15
        r_prob = .30
        a_prob = .70
    else:
        prog_len = random.randint(7, max_len - 2)
        l_prob = .05
        r_prob = .10
        a_prob = .55


    while count < prog_len:
        rand = random.random()
        if rand <= l_prob:
            add_com = '<'
        if rand > l_prob and rand <= r_prob:
            add_com = '>'
        if rand > r_prob and rand <= a_prob:
            add_com = '+'
        elif rand > a_prob:
            add_com = '-'
        prog = prog + add_com
        count += 1

    if max_len == 0:
        return prog

    low = random.randint(2, 5)
    high = random.randint(prog_len-low, prog_len)

    if high > prog_len-1:
        return prog[0:low] + '[' + prog[low:high] + ']'

    prog = prog[0:low] + '[' + prog[low:high] + ']' + prog[high:]

    return prog


def by_prob(indv: Indv) -> int:
    return indv.prob


def fill_selector(pop: list, fe: FitnessEvaluator) -> list:
    selector: list = []
    pop_score = indv_score = 0

    #find the overall population score
    for indv in pop:
        try:
            indv_score = fe.evaluate(indv)

            if indv_score == 0:
                selector.clear()
                temp = Indv(indv, indv_score)
                temp.prob = -1
                selector.append(temp)
                return selector
            selector.append(Indv(indv, indv_score))
            pop_score += indv_score
        except RuntimeError:
            pass

    #store the total score to be used for weighting later
    for s in selector:
        s.prob = pop_score - s.score

    selector.sort(reverse=True, key=by_prob)

    return selector

def before_both(cross_pt: int, loop_st1: int, loop_st2: int) -> bool:
    return cross_pt < loop_st1 and cross_pt < loop_st2

def in_both(cross_pt: int, loop_st1: int, loop_st2: int,
    loop_end1: int, loop_end2: int) -> bool:
    return (cross_pt > loop_st1 and cross_pt > loop_st2) and \
            (cross_pt < loop_end1 and cross_pt < loop_end2) and \
            (loop_end2-loop_st1 > 3 and loop_end1-loop_st2 > 3)

def after_both(cross_pt: int, loop_end1: int, loop_end2: int) -> bool:
    return cross_pt > loop_end1 and cross_pt > loop_end2

def valid(cross_pt: int, prog1: str, prog2: str, max_len: int) -> bool:
    if max_len == 0:
        return True
    try:
        loop_st1 = prog1.index('[')
        loop_st2 = prog2.index('[')
        loop_end1 = prog1.index(']')
        loop_end2 = prog2.index(']')
    except ValueError:
        return True

    return before_both(cross_pt, loop_st1, loop_st2) or \
        in_both(cross_pt, loop_st1, loop_st2, loop_end1, loop_end2) or \
        after_both(cross_pt, loop_end1, loop_end2)

#boolean function to ensure at least 3 ops are in the loop
def valid_delete(indv: str, rand_idx: int) -> bool:
    loop_st = loop_end = 0

    try:
        loop_st = indv.index('[')
        loop_end = indv.index(']')
        if (rand_idx > loop_st and rand_idx < loop_end) and \
            loop_end-loop_st < 4:
            return False
        if rand_idx in (loop_st, loop_end):
            return False
        return True
    except ValueError:
        return True

def mutate(indv: str, max_len: int) -> str:
    indv_len = len(indv)

    rand_mut = random.random()
    rand_com = random.random()
    rand_idx = random.randint(0, indv_len-1)

    if rand_com <= .15:
        add_com = '<'
    if rand_com > .15 and rand_com <= .30:
        add_com = '>'
    if rand_com > .30 and rand_com <= .65:
        add_com = '+'
    elif rand_com > .65:
        add_com = '-'



    if rand_mut <= .30 and (indv_len < max_len or max_len == 0):
        #add command
        return indv[0:rand_idx] + add_com + indv[rand_idx:]
    if rand_mut <= .60 and rand_mut > .30 and indv_len > 3:
        #remove command
        rand_idx = random.randint(2, indv_len-1)
        if valid_delete(indv, rand_idx):
            return indv[0:rand_idx] + indv[rand_idx+1:]
        return indv
    else:
        #edit command
        if indv[rand_idx] not in "[]":
            return indv[0:rand_idx] + add_com + indv[rand_idx+1:]
        return indv
        
def good_prog(indv: str, next_gen: set) -> bool:
    try:
        loop_st = indv.index('[')
        loop_end = indv.index(']')
        if loop_end - loop_st < 6:
            return False
    except ValueError:
        pass
    return True

def crossover(selector: list, pop_size: int, max_len: int, 
    no_improve: int) -> list:
    next_gen: set = set()
    candidates: list = []
    retry = num_muts = 0
    weights = [indv.prob for indv in selector]
    tog_mut = no_improve > 20

    #keep the 20 best programs no matter what
    for indv in selector[:int(pop_size*.15)]:
        if tog_mut:
            next_gen.add(mutate(indv.prog, max_len))
        else:
            next_gen.add(indv.prog)

    while len(next_gen) < pop_size:
        candidates = random.choices(selector[:int(pop_size*.4)], 
            weights[:int(pop_size*.4)], k=2)

        p1_len = len(candidates[0].prog)
        p2_len = len(candidates[1].prog)

        cross_pt = random.randint(1, min(p1_len-1, p2_len-1))

        retry = 0
        while not valid(cross_pt, candidates[0].prog, candidates[1].prog, 
            max_len):
            cross_pt = random.randint(1, min(p1_len-1, p2_len-1))
            retry += 1

            #there is likely not a valid cross point
            if retry > max_len:
                break

        #try a different combo
        if retry > max_len:
            continue

        cross1 = candidates[0].prog[0: cross_pt] + candidates[1].prog[cross_pt:]
        cross2 = candidates[1].prog[0: cross_pt] + candidates[0].prog[cross_pt:]

        #mutate each a random number of times
        num_muts = random.choices([1, 2, 3, 3, 3, 4, 5, 7], k=1)[0]

        for _ in range(num_muts):
            cross1 = mutate(cross1, max_len)
            cross2 = mutate(cross2, max_len)

        if good_prog(cross1, next_gen):
            next_gen.add(cross1)

        if len(next_gen) < pop_size and good_prog(cross2, next_gen):
            next_gen.add(cross2)

    return list(next_gen)
        


def create_program(fe: FitnessEvaluator, max_len: int) -> str:
    """
    Return a program string no longer than max_len that, when interpreted,
    populates a memory array that exactly matches a target array.

    Use fe.evaluate(program) to get a program's fitness score (zero is best).
    """
    selector: list = []
    pop_size = 75
    no_improve = 0
    prev_best = 0

    #generate the initial population of sequences
    population = init_population(max_len, pop_size)

    while True:
        selector = fill_selector(population, fe)
        #print("best prog:", selector[0].prog)
        #print("score:", selector[0].score)

        if selector[0].score == prev_best:
            no_improve += 1
        else:
            prev_best = selector[0].score
            no_improve = 0
        #we found a prog that had fitness score of 0
        if selector[0].prob == -1:
            return selector[0].prog

        population = crossover(selector, pop_size, max_len, no_improve)

    return ""



def main() -> None:  # optional driver
    array = (0, 20)
    max_len = 15

    #array = (-1, 2, 4, -5, 8, 1, 1, 3)
    #max_len = 0
    
    program = create_program(FitnessEvaluator(array), max_len)

    assert len(program) <= max_len or max_len == 0
    assert array == FitnessEvaluator.interpret(program, len(array))
    #pass



if __name__ == "__main__":
    main()
