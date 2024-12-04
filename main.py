# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import copy
import re


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

def read_lists(filename):
    lists = []
    with open(filename) as f:
        while True:
            s = f.readline()
            if s == '':
                break
            numbers = s.split()
            numbers = [int(i) for i in numbers]
            if len(numbers) > 1:
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
    print(distance)

    print(np.count_nonzero(np.unique(np1)))
    print(np.count_nonzero(np.unique(np2)))

    prev_num = -1
    prev_total = 0
    total = 0
    factors = 0
    l_idx = 0

    for num in np1:
        # If the current number matches the previous, use the previously computed total
        if num == prev_num:
            total += prev_total
            if prev_total > 0:
                print(prev_total)
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
            print(factor)
            factors += factor
            print(prev_total)

        # Update the previous number
        prev_num = num

    print(factors)
    print(total)


def day2():

    lists = read_lists('input3.txt')

    print(len(lists))

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

    print(tokens)

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

    s = read_all('input4.txt')

    line_length = s.index('\n')
    print(line_length)

    count = 0



    return count


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print(day1())
    # print(day2())
    # print(day3())
    print(day4())



# See PyCharm help at https://www.jetbrains.com/help/pycharm/













