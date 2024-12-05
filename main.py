# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import copy
import re
from dataclasses import dataclass
from collections import defaultdict
import random


def read(filename):
    arr1 = []
    arr2 = []
    with open(filename) as f:
        while True:
            s = f.readline()
            if s == '':
                break
            num1, num2 = s.split()
            arr1.append(int(num1))
            arr2.append(int(num2))

    return arr1, arr2


def read_lists(filename, is_int=True):
    lists = []
    with open(filename) as f:
        while True:
            s = f.readline()
            if s == '':
                break
            numbers = s.split()
            if is_int:
                numbers = [int(i) for i in numbers]
            lists.append(numbers)

    return lists

def read_all(filename):
    with open(filename) as f:
        s = f.read()
        return s


def day1():

    arr1, arr2 = read('input1.txt')
    arr1.sort(), arr2.sort()
    np1 = np.array(arr1)
    np2 = np.array(arr2)

    distance = np.sum(np.abs(np.subtract(np1, np2)))

    prev_num = -1
    prev_total = 0
    total = 0
    factors = 0
    l_idx = 0

    for num in np1:
        # If the current number matches the previous, use the previously computed total
        if num == prev_num:
            total += prev_total
            continue

        # Reset factor to count occurrences for the current num
        factor = 0
        # Count occurrences of num in np2, ensuring l_idx does not go out of bounds
        while l_idx < len(np2) and num >= np2[l_idx]:
            if num == np2[l_idx]:
                factor += 1
            l_idx += 1

        # Calculate the total for the current number
        prev_total = factor * num
        total += prev_total

        if prev_total > 0:
            factors += factor

        # Update the previous number
        prev_num = num
    return total


def day2():

    lists = read_lists('input2.txt')

    safe = 0
    for orig_lst in lists:
        at_least_one_safe = False
        for idx in range(len(orig_lst)):
            lst = copy.copy(orig_lst)
            lst.pop(idx)
            list_idx = 1
            trend_up = lst[0] < lst[1]
            while list_idx < len(lst):
                diff = abs(lst[list_idx - 1] - lst[list_idx])
                if diff > 3 or diff < 1:
                    break
                is_up = lst[list_idx - 1] < lst[list_idx]
                if is_up != trend_up:
                    break
                list_idx += 1
            else:
                at_least_one_safe = True
        if at_least_one_safe:
            safe += 1

    return safe


def day3():

    input = read_all('input3.txt')

    # Define token patterns
    token_specification = [
        ('MUL',  r'mul'),
        ('NUMBER', r'\d+(\.\d*)?'),  # Integer or decimal number
        ('COMMA', r','),
        ('LPAREN', r'\('),  # Left parenthesis
        ('RPAREN', r'\)'),  # Right parenthesis
        ('NT', r'n\'t'),
        ('DO', r'do'),
        ('X', r'.'),  # Any other character
    ]

    # Compile the regex patterns into a single pattern
    token_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_specification)

    # Example string to tokenize
    # input = 'xmul(2,4)&mul[3,7]!^don\'t()_mul(5,5)+mul(32,64](mul(11,8)undo()?mul(8,5))'

    # Tokenize
    tokens = []
    for mo in re.finditer(token_regex, input):
        kind = mo.lastgroup
        value = mo.group()
        if kind == 'NUMBER':
            value = float(value) if '.' in value else int(value)

        tokens.append((kind, value))

    # 174199222
    # 173731097

    total = 0

    do = True

    i = 0
    while i < len(tokens) - 5:
        if tokens[i][0] == 'MUL' and \
                tokens[i + 1][0] == 'LPAREN' and \
                tokens[i + 2][0] == 'NUMBER' and \
                tokens[i + 3][0] == 'COMMA' and \
                tokens[i + 4][0] == 'NUMBER' and \
                tokens[i + 5][0] == 'RPAREN':
            if do:
                total += tokens[i + 2][1] * tokens[i + 4][1]
            i += 6  # Skip the matched tokens
        elif tokens[i][0] == 'DO' and \
                tokens[i + 1][0] == 'LPAREN' and \
                tokens[i + 2][0] == 'RPAREN':
            do = True
            i += 3
        elif tokens[i][0] == 'DO' and \
                tokens[i + 1][0] == 'NT' and \
                tokens[i + 2][0] == 'LPAREN' and \
                tokens[i + 3][0] == 'RPAREN':
            do = False
            i += 4
        else:
            i += 1

    return total


def day4():

    grid = read_lists('input4.txt', is_int=False)
    grid = [list(line[0]) for line in grid]

    line_length = len(grid[0])
    lines_count = len(grid)
    count = 0

    @dataclass
    class Coor:
        row: int = 0
        col: int = 0

        def __add__(self, other):
            n_row = self.row + other.row
            n_col = self.col + other.col
            return Coor(n_row, n_col)

        def __sub__(self, other):
            n_row = self.row - other.row
            n_col = self.col - other.col
            return Coor(n_row, n_col)

    def check_adjacent_(letter, direction, c: Coor):
        n = c + direction
        if 0 <= n.row < lines_count and 0 <= n.col < line_length:
            return grid[n.row][n.col] == letter, n
        return False, n

    to_match = 'MAS'
    directions = [Coor(vert, hori) for vert in [-1, 1] for hori in [-1, 1]]

    coordinate_dict = {}

    for i, row in enumerate(grid):
        for j, ele in enumerate(row):
            if ele == to_match[0]:
                for direction in directions:
                    c = Coor(i, j)
                    for idx in range(1, len(to_match)):
                        found, next_c = check_adjacent_(to_match[idx], direction, c)
                        if not found:
                            break
                        elif idx == len(to_match) - 1:
                            coordinate_tuple = (c.row, c.col)
                            if coordinate_tuple in coordinate_dict:
                                coordinate_dict[coordinate_tuple] += 1
                            else:
                                coordinate_dict[coordinate_tuple] = 1
                        c = next_c
    return len([cnt for coord, cnt in coordinate_dict.items() if cnt > 1])


def day5():

    input_str = read_all('input5.txt')

#     input_str = """47|53
# 97|13
# 97|61
# 97|47
# 75|29
# 61|13
# 75|53
# 29|13
# 97|29
# 53|29
# 61|53
# 97|53
# 61|29
# 47|13
# 75|47
# 97|75
# 47|61
# 75|61
# 47|29
# 75|13
# 53|13
#
# 75,47,61,53,29
# 97,61,53,29,13
# 75,29,13
# 75,97,47,61,53
# 61,13,29
# 97,13,75,29,47"""

    sections = input_str.split('\n\n')

    rules_list = sections[0].split('\n')
    lists = sections[1].split('\n')

    rules_list = [tuple(map(int, r.split('|'))) for r in rules_list]
    lists = [[int(num) for num in i.split(',') if num.strip()] for i in lists]

    print(lists[0])

    # print(rules[-1])

    rules_list.sort()

    rules = defaultdict(list)

    for a, b in rules_list:
        rules[a].append(b)

    count = 0

    def is_valid(manual):
        violations = []
        for i, num in enumerate(manual):
            # find the rules of num
            for greater in rules[num]:
                if greater in manual[:i]:
                    j = manual[:i].index(greater)
                    violations.append((i, j))
        return violations

    for manual in lists:
        violations = is_valid(manual)
        # # part 1
        # if violations == 0 and len(manual) > 0:
        #     count += manual[int((len(manual)-1)/2)]
        pass

    # Part 2
    dependencies = defaultdict(set)
    in_degree = defaultdict(int)
    all_elements = set()

    for a, b in rules_list:
        dependencies[a].add(b)
        in_degree[b] += 1
        all_elements.update([a, b])

    for element in all_elements:
        if element not in in_degree:
            in_degree[element] = 0

    ordering = []
    queue = [node for node in in_degree if in_degree[node] == 0]

    while queue:
        current = queue.pop(0)
        ordering.append(current)
        for neighbor in dependencies[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If there are elements not included due to cycles, add them in an arbitrary consistent way
    for element in all_elements:
        if element not in ordering:
            ordering.append(element)

    # Create an order map to use for sorting
    order_map = {num: index for index, num in enumerate(ordering)}

    def sort_list(lst, order_map):
        return sorted(lst, key=lambda x: order_map.get(x, 0))

    # Correct a list and find the middle number
    # lists = [[i for i in range(100)]]
    for manual in lists:
        corrected = copy.copy(manual)
        iterations = 0
        while True:
            iterations += 1
            violations = is_valid(corrected)
            if len(violations) == 0:
                break
            elif iterations % 100 == 0:
                random.shuffle(corrected)
            for i, j in violations:
                corrected[i], corrected[j] = corrected[j], corrected[i]
        count += corrected[int((len(corrected) - 1) / 2)]
    return count


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print(day1())
    # print(day2())
    # print(day3())
    # print(day4())
    print(day5())




# See PyCharm help at https://www.jetbrains.com/help/pycharm/













