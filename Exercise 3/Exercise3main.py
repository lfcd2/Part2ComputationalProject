from Exercise3part1 import run_ex3p1
from Exercise3part2 import run_ex3p2a, run_ex3p2b
from Exercise3GeneralCase import run_ex3gc


def get_input(options) -> str:
    inp = input("""Which part of the code do you want to run?
A: Part 1
B: Part 2.1
C: Part 2.2
D: General Case
Please enter the relevant letter: """)
    if inp.upper() not in options:
        print("Invalid input, try again")
        inp = get_input(options)
    return inp


def run():
    options = ['A', 'B', 'C', 'D']
    inp = get_input(options)
    funcs = [run_ex3p1, run_ex3p2a, run_ex3p2b, run_ex3gc]
    function_to_run = funcs[options.index(inp)]
    function_to_run()


if __name__ == '__main__':
    run()
