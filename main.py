# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import copy
import re
from dataclasses import dataclass
from collections import defaultdict, deque
import random
from functools import total_ordering
from itertools import combinations
import time


def printt(function, start):
    to_print = function()
    end = time.process_time()
    elapsed = end-start
    print(str(to_print) + '   ' + str(round(elapsed, 3)))
    return end


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


@dataclass(frozen=True)
class Coor:
    __slots__ = ('row', 'col')  # Manually define slots
    row: int
    col: int

    def __add__(self, other):
        n_row = self.row + other.row
        n_col = self.col + other.col
        return Coor(n_row, n_col)

    def __sub__(self, other):
        n_row = self.row - other.row
        n_col = self.col - other.col
        return Coor(n_row, n_col)

def check_adjacent_(letter, grid, direction, c: Coor, lines_count, line_length):
    n = c + direction
    if 0 <= n.row < lines_count and 0 <= n.col < line_length:
        return grid[n.row][n.col] == letter, n
    return False, n


def day4():

    grid = read_lists('input4.txt', is_int=False)
    grid = [list(line[0]) for line in grid]

    line_length = len(grid[0])
    lines_count = len(grid)
    count = 0

    to_match = 'MAS'
    directions = [Coor(vert, hori) for vert in [-1, 1] for hori in [-1, 1]]

    coordinate_dict = {}

    for i, row in enumerate(grid):
        for j, ele in enumerate(row):
            if ele == to_match[0]:
                for direction in directions:
                    c = Coor(i, j)
                    for idx in range(1, len(to_match)):
                        found, next_c = check_adjacent_(to_match[idx], grid, direction, c, lines_count, line_length)
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

    # print(lists[0])

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

    for manual in lists:
        if len(is_valid(manual)) == 0:
            continue
        manual_rules = {
            rule: {num for num in rules[rule] if num in manual}
            for rule in rules if rule in manual
        }

        in_degree = defaultdict(int)

        for ele in manual:
            in_degree[ele] = 0

        for rule in manual_rules:
            for b in manual_rules[rule]:
                in_degree[b] += 1

        queue = [ele for ele in in_degree if in_degree[ele] == 0]
        ordering = []

        while queue:
            current = queue.pop(0)
            ordering.append(current)
            if current in manual_rules:  # Ensure current has dependencies
                for neighbor in manual_rules[current]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        # If there are elements not included due to cycles, add them in an arbitrary consistent way
        for element in manual:
            if element not in ordering:
                ordering.append(element)
                # print('dingdong')

        # print(len(ordering))
        count += ordering[int((len(ordering)-1)/2)]

    return count


def day6():

    string = read_all('input6.txt')
    rows = string.split('\n')

    map = np.zeros((len(rows), len(rows[0])), int)
    visited = np.zeros_like(map, int)
    # print(visited.shape)

    for i, row in enumerate(rows):
        for j, char in enumerate(row):
            if char == '.':
                map[i, j] = 0
            elif char == '#':
                map[i, j] = 1
            elif char == '^':
                map[i, j] = 2

    # print(np.argwhere(map == 2)[0])


def day7():
    pass


def day8():


    # def BFS(graph, node):
    #     visited = []  # List for visited nodes.
    #     queue = []  # Initialize a queue
    #     visited.append(node)
    #     queue.append(node)
    #     while queue:  # Creating loop to visit each node
    #         m = queue.pop(0)
    #         print(m, end=" ")
    #         for direction in cardinal_dir[m]:
    #
    #             if neighbour not in visited:
    #                 visited.append(neighbour)
    #                 queue.append(neighbour)

    lists = read_all('input8.txt')
#     lists = """............
# ........0...
# .....0......
# .......0....
# ....0.......
# ......A.....
# ............
# ............
# ........A...
# .........A..
# ............
# ............"""

    grid = [list(row) for row in lists.splitlines()]

    unique_towers = {ele for row in grid for ele in row} - {'.'}
    # print(unique_towers)

    def is_in_bounds(coord, dims):
        return 0 <= coord.row < dims[0] and 0 <= coord.col < dims[0]

    all_towers = defaultdict(list)
    for row_idx, row in enumerate(grid):
        for col_idx, ele in enumerate(row):
            if ele != '.':
                all_towers[ele].append(Coor(row_idx, col_idx))

    all_towers = dict(all_towers)

    interference_patterns = {}
    for tower_type, coordinates in all_towers.items():
        sorted_coordinates = sorted(coordinates, key=lambda c: (c.row, c.col))
        interference_patterns[tower_type] = list(combinations(sorted_coordinates, 2))

    dims = (len(grid), len(grid[0]))
    nodes = np.zeros(dims)

    for tower_type, patterns in interference_patterns.items():
        for pattern in patterns:
            direction = pattern[1] - pattern[0]
            antinode1 = pattern[1]
            antinode2 = pattern[0]
            while(is_in_bounds(antinode1, dims)):
                nodes[antinode1.row, antinode1.col] = 1
                antinode1 = antinode1 + direction
            while (is_in_bounds(antinode2, dims)):
                nodes[antinode2.row, antinode2.col] = 1
                antinode2 = antinode2 - direction

    for row_idx, row in enumerate(grid):
        new_row = []
        for col_idx, ele in enumerate(row):
            if ele == '.' and nodes[row_idx, col_idx] == 1:
                new_row.append('#')
            else:
                new_row.append(ele)
        # print(''.join(new_row))

    return int(np.sum(nodes))


def day9():

    in_file = read_all('input9.txt')
    # in_file = '2333133121414131402'

    nums_list = [int(num) for num in list(in_file) if num != '\n']

    file_blocks = []
    for i, ele in enumerate(nums_list):
        if i % 2 == 0:
            for _ in range(ele):
                file_blocks.append(i//2)
        else:
            for _ in range(ele):
                file_blocks.append('.')

    file_blocks_2_alloc = []
    file_blocks_2_free = []
    for i, ele in enumerate(nums_list):
        if i % 2 == 0:
            file_blocks_2_alloc.append((i, i//2, ele))
            file_blocks_2_free.append(0)
        else:
            file_blocks_2_free.append(ele)

    nums = deque(file_blocks)

    # print(nums)

    FINAL = []
    while nums:
        left = nums.popleft()
        if left == '.':
            right = '.'
            while right == '.':
                right = nums.pop()
            FINAL.append(right)
        else:
            FINAL.append(left)

    # print(FINAL)

    checksum = 0
    for i, ele in enumerate(FINAL):
        checksum += i*ele

    # print(file_blocks_2_alloc)

    FINAL_2 = []
    while file_blocks_2_alloc:
        last = file_blocks_2_alloc.pop()
        index_to_swap = None
        for i, space in enumerate(file_blocks_2_free):
            if i > last[0]:
                break
            if space >= last[2]:
                index_to_swap = i
                break
        if index_to_swap is not None:
            pos, id, count = last
            file_blocks_2_free[index_to_swap] -= count
            file_blocks_2_free[pos] = count
            FINAL_2.append((index_to_swap, id, count))
        else:
            FINAL_2.append(last)

    for pos, space in enumerate(file_blocks_2_free):
        FINAL_2.append((pos, -1, space))

    FINAL_2 = sorted(FINAL_2, key = lambda x: (x[0],-x[1]))

    # print(FINAL_2)

    out = []
    for i, id, count in FINAL_2:
        to_print = '.' if id == -1 else id
        for _ in range(count):
            out.append(to_print)

    checksum_2 = 0
    for i, id in enumerate(out):
        # print(id, end='')
        if id != '.':
            checksum_2 += i*id
    # print('')

    return checksum_2


def day10():

    grid = read_all('input10.txt')
#     grid = """89010123
# 78121874
# 87430965
# 96549874
# 45678903
# 32019012
# 01329801
# 10456732"""

    grid = [[int(char) for char in row] for row in grid.splitlines()]
    grid_height = len(grid)
    grid_width = len(grid[0])

    # print(grid)

    # how many valid trails are reachable
    reachable_ends = np.zeros((grid_width, grid_height))
    trail_signature = np.zeros((grid_width, grid_height))

    directions = [Coor(vert, hori) for vert in [-1, 0, 1] for hori in [-1, 0, 1] if abs(vert) != abs(hori)]
    # print(directions)

    # init_coor = [(Coor(0, 0), [])]
    trails = defaultdict(list)
    trails_rev = defaultdict(list)
    visited = np.zeros((grid_width, grid_height))

    for i, row in enumerate(grid):
        for j, num in enumerate(row):
            # start a trail
            for dir in directions:
                current = Coor(i, j)
                found, new = check_adjacent_(num+1, grid, dir, current, grid_height, grid_width)
                if found:
                    trails[current].append(new)
                found_rev, new_rev = check_adjacent_(num-1, grid, dir, current, grid_height, grid_width)
                if found_rev:
                    trails_rev[current].append(new_rev)

    def follow_rev(pos: Coor):
        if visited[pos.row, pos.col]:
            return
        # print('new at  ' + str(pos))
        visited[pos.row, pos.col] = True
        reachable_ends[pos.row, pos.col] += 1
        return [follow_rev(trail) for trail in trails_rev[pos]]

    for i, row in enumerate(grid):
        for j, num in enumerate(row):
            if num == 9:
                visited = np.zeros((grid_width, grid_height))
                follow_rev(Coor(i, j))

    def follow(pos: Coor):
        if visited[pos.row, pos.col]:
            return trail_signature[pos.row, pos.col]
        visited[pos.row, pos.col] = True
        if grid[pos.row][pos.col] == 9:
            trail_signature[pos.row, pos.col] = 1
        else:
            trail_signature[pos.row, pos.col] = sum([follow(trail) for trail in trails[pos]])
        return trail_signature[pos.row, pos.col]

    count = 0
    visited = np.zeros((grid_width, grid_height))
    for i, row in enumerate(grid):
        for j, num in enumerate(row):
            if num == 0:
                # count += reachable_ends[i, j]
                count += follow(Coor(i, j))

    return count




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    t = time.process_time()

    # t = printt(day1, t)
    # t = printt(day2, t)
    # t = printt(day3, t)
    # t = printt(day4, t)
    # t = printt(day5, t)
    # t = printt(day6, t)
    # t = printt(day7, t)
    # t = printt(day8, t)
    # t = printt(day9, t)
    t = printt(day10, t)
    # t = printt(day11, t)
    # t = printt(day12, t)
    # t = printt(day13, t)
    # t = printt(day14, t)
    # t = printt(day15, t)
    # t = printt(day16, t)
    # t = printt(day17, t)
    # t = printt(day18, t)
    # t = printt(day19, t)





# See PyCharm help at https://www.jetbrains.com/help/pycharm/













