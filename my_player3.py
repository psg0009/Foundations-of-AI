"""
CSCI-561 Fall 2026 - Homework 2 Parth Gosar (9609181412)
Little-Go 5x5: negamax with alpha-beta pruning, iterative deepening and a transposition table.

"""

import json
import time

_clock = time.process_time

MOVE_BUDGET = 3.0

SIZE = 5
LIMIT = 24          # a 5x5 game is cut off after (5*5)-1 moves in total
PASS = -1
KOMI = 2.5          # score compensation awarded to White for moving second
STATE_FILE = "agent_state.json"



# Board geometry

NEIGHBORS = []
for position in range(25):
    row, column = divmod(position, SIZE)
    adjacent_points = (
        (row - 1, column), (row + 1, column),
        (row, column - 1), (row, column + 1),
    )
    NEIGHBORS.append(tuple(
        next_row * SIZE + next_column
        for next_row, next_column in adjacent_points
        if 0 <= next_row < SIZE and 0 <= next_column < SIZE
    ))



# I/O

def read_input(path="input.txt"):
    with open(path) as handle:
        lines = [line.strip() for line in handle if line.strip()]
    color = int(lines[0])
    previous = tuple(int(cell) for row in lines[1:6] for cell in row)
    board = tuple(int(cell) for row in lines[6:11] for cell in row)
    return color, previous, board


def write_output(move, path="output.txt"):
    if move == PASS:
        text = "PASS"
    else:
        row, column = divmod(move, SIZE)
        text = "{},{}".format(row, column)
    with open(path, "w", newline="\n") as handle:
        handle.write(text + "\n")



# Rules

class SearchTimeout(Exception):
    pass


def group_and_liberties(board, start):
    color = board[start]
    group, liberties, pending = ({start}, set(), [start])
    while pending:
        position = pending.pop()
        for neighbor in NEIGHBORS[position]:
            if board[neighbor] == 0:
                liberties.add(neighbor)
            elif board[neighbor] == color and neighbor not in group:
                group.add(neighbor)
                pending.append(neighbor)
    return (group, liberties)


def play(board, previous, color, move):
    if move == PASS:
        return (board, 0)
    if board[move] != 0:
        return None
    result = list(board)
    result[move] = color
    checked_stones, captured = (set(), set())
    for neighbor in NEIGHBORS[move]:
        if result[neighbor] == 3 - color and neighbor not in checked_stones:
            group, liberties = group_and_liberties(result, neighbor)
            checked_stones.update(group)
            if not liberties:
                captured.update(group)
    for neighbor in captured:
        result[neighbor] = 0
    if not group_and_liberties(result, move)[1]:
        return None
    result = tuple(result)
    # Ko forbids immediately recreating the board before the opponent moved
    if result == previous:
        return None
    return (result, len(captured))



# Evaluation

def score_margin(board, color):
    margin = board.count(1) - board.count(2) - KOMI
    return margin if color == 1 else -margin


def evaluate(board, color):
    # Only stones count toward the final score; empty territory does not.
    value = 10.0 * score_margin(board, color)
    visited = set()
    for position in range(25):
        if not board[position] or position in visited:
            continue
        group, liberties = group_and_liberties(board, position)
        visited.update(group)
        sign = 1 if board[position] == color else -1
        liberty_count = len(liberties)
        value += sign * 1.5 * min(liberty_count, 6)
        # A group with one liberty can be captured on the next move, and one with two is a move away from that, so both are graded by group size.
        if liberty_count == 1:
            value -= sign * (8.0 + 5.0 * len(group))
        elif liberty_count == 2:
            value -= sign * 1.5 * len(group)
    for position, stone in enumerate(board):
        if stone:
            row, column = divmod(position, SIZE)
            sign = 1 if stone == color else -1
            value += sign * 0.3 * (4 - abs(row - 2) - abs(column - 2))
    return value


def terminal_value(board, color):
    margin = score_margin(board, color)
    return (100000.0 if margin > 0 else -100000.0) + margin



# Search

class Agent:
    def __init__(self, seconds=MOVE_BUDGET):
        self.deadline = _clock() + seconds
        self.transpositions = {}

    def check_time(self):
        if _clock() >= self.deadline:
            raise SearchTimeout()

    def ordered_moves(self, board, previous, color, preferred=None):
        candidates = []
        for move in range(25):
            self.check_time()
            if board[move]:
                continue
            outcome = play(board, previous, color, move)
            if outcome is None:
                continue
            next_board, captures = outcome
            priority = evaluate(next_board, color) + 12.0 * captures
            if move == preferred:
                priority += 1000000.0
            candidates.append((priority, move, next_board))
        priority = evaluate(board, color) - 3.0
        if preferred == PASS:
            priority += 1000000.0
        candidates.append((priority, PASS, board))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates

    def search(self, board, previous, color, depth, alpha, beta, moves_played, passes):
        self.check_time()
        if moves_played >= LIMIT or passes >= 2:
            return terminal_value(board, color)
        if depth == 0:
            return evaluate(board, color)
        # The previous board matters for ko, and the move count affects endings.
        key = (board, previous, color, moves_played, passes)
        entry = self.transpositions.get(key)
        preferred = None
        original_alpha, original_beta = (alpha, beta)
        if entry is not None:
            stored_depth, stored_value, bound_type, preferred = entry
            if stored_depth >= depth:
                if bound_type == 'EXACT':
                    return stored_value
                if bound_type == 'LOWER':
                    alpha = max(alpha, stored_value)
                else:
                    beta = min(beta, stored_value)
                if alpha >= beta:
                    return stored_value
        best_value, best_move = (float('-inf'), PASS)
        for _, move, next_board in self.ordered_moves(board, previous, color, preferred):
            next_passes = passes + 1 if move == PASS else 0
            value = -self.search(
                next_board, board, 3 - color, depth - 1,
                -beta, -alpha, moves_played + 1, next_passes,
            )
            if value > best_value:
                best_value, best_move = (value, move)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        if best_value <= original_alpha:
            bound_type = 'UPPER'
        elif best_value >= original_beta:
            bound_type = 'LOWER'
        else:
            bound_type = 'EXACT'
        self.transpositions[key] = (depth, best_value, bound_type, best_move)
        return best_value

    def choose(self, board, previous, color, moves_played, passes):
        # Keeping a legal fallback in case the search runs out of time.
        chosen_move = PASS
        for move in range(25):
            if play(board, previous, color, move) is not None:
                chosen_move = move
                break
        try:
            root_moves = self.ordered_moves(board, previous, color)
        except SearchTimeout:
            return chosen_move
        max_depth = max(1, LIMIT - moves_played)
        for depth in range(1, max_depth + 1):
            best_move, best_value, alpha = (chosen_move, float('-inf'), float('-inf'))
            try:
                root_moves.sort(key=lambda item: item[1] == chosen_move, reverse=True)
                for _, move, next_board in root_moves:
                    self.check_time()
                    next_passes = passes + 1 if move == PASS else 0
                    value = -self.search(
                        next_board, board, 3 - color, depth - 1,
                        float('-inf'), -alpha, moves_played + 1, next_passes,
                    )
                    if value > best_value:
                        best_value, best_move = (value, move)
                    alpha = max(alpha, value)
            except SearchTimeout:
                break
            # An interrupted depth may have examined only some moves. 
            # Update the answer only after finishing the whole depth.
            chosen_move = best_move
            if depth >= LIMIT - moves_played:
                break
        return chosen_move



# Driver

def recover_progress(color, previous, board):
    if not any(previous):
        if not any(board):
            # Two empty boards mean either the opening move as Black, or that black opened with a pass. 
            # In the second case this turn is the second consecutive pass if White passes too, which ends the game and wins it on komi alone, so the pass has to be visible here.
            return (0, 0) if color == 1 else (1, 1)
        if color == 2 and board.count(1) == 1 and (board.count(2) == 0):
            return (1, 0)
    try:
        with open(STATE_FILE) as handle:
            saved = json.load(handle)
        if saved['color'] == color and tuple(saved['after_board']) == previous:
            move_count = int(saved['moves_played']) + 1
            passes = int(saved['passes']) + 1 if board == previous else 0
            return (move_count, passes)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    occupied = sum((stone != 0 for stone in board))
    if occupied % 2 != (0 if color == 1 else 1):
        occupied += 1
    return (occupied, int(board == previous))


def save_progress(color, after_board, moves_played, passes):
    saved = {
        'color': color,
        'after_board': list(after_board),
        'moves_played': moves_played,
        'passes': passes,
    }
    try:
        with open(STATE_FILE, "w") as handle:
            json.dump(saved, handle)
    except OSError:
        pass


def main():
    color, previous, board = read_input()
    moves_played, passes = recover_progress(color, previous, board)
    move = Agent().choose(board, previous, color, moves_played, passes)
    outcome = play(board, previous, color, move)
    if outcome is None:
        move, after_board = (PASS, board)
    else:
        after_board = outcome[0]
    write_output(move)
    save_progress(color, after_board, moves_played + 1,
                  passes + 1 if move == PASS else 0)


if __name__ == '__main__':
    main()
