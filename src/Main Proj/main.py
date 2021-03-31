from collections import deque
from state import State
from pieces import Piece
from copy import deepcopy
import itertools
import levels
from heapq import heappush, heappop, heapify

import time
import resource
from memory_profiler import memory_usage

initial_state = list()

max_frontier_size = 0
nodes_expanded = 0
max_search_depth = 1

def print_board(board):
    side_len = int(len(board) ** 0.5)
    for i in range(side_len):
        for j in range(side_len):
            print(board[i * side_len + j] + " ", end="")
        print()

# Prints a list of tuples of strings as a nice table
def print_table(table):
    col_width = [max(len(x) for x in col) for col in zip(*table)]
    for line in table:
        print ("| " + " | ".join("{:{}}".format(x, col_width[i])
                                for i, x in enumerate(line)) + " |")


# Checks if every movable piece has reached its destination
def check_end(pieces):
    for i in range(len(pieces)):
        # accesses tuple on same pos of movable and destination arrays and compares x and y coords
        if (pieces[i].check_coords_inequality()):
            return False

    return True

def bfs(start_state, pieces):

    global goal_node, max_search_depth
    
    explored, queue = set(), deque([State(start_state, None, "", 0, 0, 0, pieces)])

    while queue:

        node = queue.popleft()
        explored.add(node.map)

        if (check_end(node.pieces)):
            print("Solution: {}".format(node.move))
            print_board(node.state)
            return node.move

        neighbours = expand(node)

        for neighbour in neighbours:
            if neighbour.map not in explored:
                queue.append(neighbour)
                if neighbour.depth > max_search_depth:
                    max_search_depth += 1

def dfs(start_state, pieces):

    global max_frontier_size, goal_node, max_search_depth

    explored, stack = set(), list([State(start_state, None, "", 0, 0, 0, pieces)])

    while stack:

        node = stack.pop()
        explored.add(node.map)

        if (check_end(node.pieces)):
            print("Solution: {}".format(node.move))
            print_board(node.state)
            return node.move

        neighbors = reversed(expand(node))

        for neighbor in neighbors:
            if neighbor.map not in explored:
                stack.append(neighbor)
                explored.add(neighbor.map)

                if neighbor.depth > max_search_depth:
                    max_search_depth += 1

        if len(stack) > max_frontier_size:
            max_frontier_size = len(stack)

def h(state):
    cost = 1

    for i in range(len(state.pieces)):
        cost += ((state.pieces[i].movable_row - state.pieces[i].dest_row)**2 + (state.pieces[i].movable_col - state.pieces[i].dest_col)**2)**1/2

    return cost

def a_star(start_state, pieces):

    global max_frontier_size, goal_node, max_search_depth

    explored, heap, heap_entry, counter = set(), list(), {}, itertools.count()

    root = State(start_state, None, "", 0, 0, 0, pieces)

    key = h(root)

    root.key = key

    entry = (key, 0, root)

    heappush(heap, entry)

    heap_entry[root.map] = entry

    while heap:

        node = heappop(heap)
        explored.add(node[2].map)
        if (check_end(node[2].pieces)):
            print("Solution: {}".format(node[2].move))
            print_board(node[2].state)
            return node[2].move

        neighbors = expand(node[2])

        for neighbor in neighbors:

            
            neighbor.key = neighbor.cost + h(neighbor)

            entry = (neighbor.key, neighbor.move, neighbor)

            if neighbor.map not in explored:

                heappush(heap, entry)

                explored.add(neighbor.map)

                heap_entry[neighbor.map] = entry

                if neighbor.depth > max_search_depth:
                    max_search_depth += 1

            elif neighbor.map in heap_entry and neighbor.key < heap_entry[neighbor.map][2].key:

                hindex = heap.index((heap_entry[neighbor.map][2].key,
                                     heap_entry[neighbor.map][2].move,
                                     heap_entry[neighbor.map][2]))

                heap[int(hindex)] = entry

                heap_entry[neighbor.map] = entry

                heapify(heap)

        if len(heap) > max_frontier_size:
            max_frontier_size = len(heap)


def iterative_deepening(start_state, pieces):
    global max_frontier_size, goal_node, max_search_depth

    current_search_depth = 1

    while True:

        explored, stack = set(), list([State(start_state, None, "", 0, 0, 0, pieces)])
        while stack:

            node = stack.pop()

            explored.add(node.map)

            if (check_end(node.pieces)):
                print("Solution: {}".format(node.move))
                print_board(node.state)
                return node.move
            

            neighbors = reversed(expand(node))

            for neighbor in neighbors:
                if neighbor.map not in explored and node.depth < current_search_depth: 
                    stack.append(neighbor)
                    explored.add(neighbor.map)

                    if neighbor.depth > max_search_depth:
                        max_search_depth += 1

            if len(stack) > max_frontier_size:
                max_frontier_size = len(stack)
        current_search_depth += 1


def expand(node):
    print("Expanding node: {} Depth: {}".format(node.move, node.depth))

    global nodes_expanded
    nodes_expanded += 1

    neighbours = list()

    if (node.move != ""):
        if (node.move[-1] == "u" or node.move[-1] == "d"):
            neighbours.append(move(node, "r"))
            neighbours.append(move(node, "l"))
        
        elif (node.move[-1] == "l" or node.move[-1] == "r"):
            neighbours.append(move(node, "u"))
            neighbours.append(move(node, "d"))
    
    else:
        neighbours.append(move(node, "u"))
        neighbours.append(move(node, "d"))
        neighbours.append(move(node, "r"))
        neighbours.append(move(node, "l"))
    
    return neighbours


def move(node, offset):

    new_node = deepcopy(node)

    sort_pieces(new_node.pieces, offset)

    for i in range(len(new_node.pieces)):
        cur_row = new_node.pieces[i].movable_row
        cur_col = new_node.pieces[i].movable_col

        if (offset == "u"):
            newCoords = moveUp(new_node.state, cur_row, cur_col)
            new_node.pieces[i].movable_row = newCoords[0]

        elif (offset == "d"):
            newCoords = moveDown(new_node.state, cur_row, cur_col)
            new_node.pieces[i].movable_row = newCoords[0]

        elif (offset == "l"):
            newCoords = moveLeft(new_node.state, cur_row, cur_col)
            new_node.pieces[i].movable_col = newCoords[1]

        elif (offset == "r"):
            # print("Row: {} Col: {}".format(cur_row, cur_col))
            newCoords = moveRight(new_node.state, cur_row, cur_col)
            # print(newCoords)
            new_node.pieces[i].movable_col = newCoords[1]
            
        size_board = int(len(new_node.state) ** 0.5)
        new_node.state[cur_row * size_board + cur_col] = "."
        new_node.state[newCoords[0] * size_board + newCoords[1]] = new_node.pieces[i].movable_symbol

    new_node.calc_map()

    new_node.parent = node
    new_node.move = new_node.parent.move + offset
    new_node.depth = new_node.parent.depth + 1
    return new_node


# -----------------------------------------

def sort_pieces(pieces, move):
    if (move == "u"):
        pieces.sort(key=lambda x: x.movable_row, reverse=False)
    elif (move == "d"):
        pieces.sort(key=lambda x: x.movable_row, reverse=True)
    elif (move == "l"):
        pieces.sort(key=lambda x: x.movable_col, reverse=False)
    elif (move == "r"):
        pieces.sort(key=lambda x: x.movable_col, reverse=True)

def moveUp(board, cur_row, cur_col):
    return getNewPiecePosition(board, cur_row, cur_col, -1, 0)

def moveDown(board, cur_row, cur_col):
    return getNewPiecePosition(board, cur_row, cur_col, 1, 0)

def moveLeft(board, cur_row, cur_col):
    return getNewPiecePosition(board, cur_row, cur_col, 0, -1)

def moveRight(board, cur_row, cur_col):
    return getNewPiecePosition(board, cur_row, cur_col, 0, 1)

def getNewPiecePosition(board, curRow, curCol, rowMov, colMov):
    size_board = int(len(board) ** 0.5)

    if (rowMov == 0 and colMov == 0): return [curRow, curCol]

    newRow = curRow
    newCol = curCol

    while (True):
        # print("Row: {} Col: {}".format(newRow, newCol))

        calc_pos = size_board * (newRow + rowMov) + newCol + colMov

        if (calc_pos >= 0 and calc_pos < len(board) and newRow + rowMov >= 0 and newRow + rowMov < size_board and newCol + colMov >= 0 and newCol + colMov < size_board):
            
            if (board[calc_pos] != "." and board[calc_pos] != "P" and board[calc_pos] != "T"):
                break # if move is to an occupied tile
            else:
                newRow += rowMov
                newCol += colMov
        else:
            break
    
    # print("Returning: {} {}".format(newRow, newCol))
    return [newRow, newCol]


# -----------------------------------------

def read_move():
    while(True):
        print("Choose your move:")
        print("up -> u | down -> d | left -> l | right -> r | undo | restart")
        move = input("> ")
        move.lower()

        if(move in ["u", "d", "l", "r", "undo", "restart"]):
            return move

        print("Illegal move!")


def player_loop():

    (board, pieces) = levels.lvl1()

    previous_board = deepcopy(board)
    previous_pieces = deepcopy(pieces)

    original_board = deepcopy(board)
    original_pieces = deepcopy(pieces)

    current_move = ""
    found_first_undo = False

    print()
    print_board(board)

    while (True):
        if (check_end(pieces)):
            print("Level Completed!")
            break

        print()
        new_move = read_move()

        if (new_move == "undo" and found_first_undo == False):
            found_first_undo = True
            board = previous_board
            pieces = previous_pieces
            current_move = current_move[:-1]

        elif (new_move == "undo"):
            print("Second Undo in a Row. not allowed :p")
            pass
            
        elif (new_move == "restart"):
            found_first_undo = False
            board = original_board
            pieces = original_pieces
            current_move = ""

        else:
            found_first_undo = False
            previous_board = deepcopy(board)
            previous_pieces = deepcopy(pieces)

            current_move += new_move

        print("Current Move Sequence: {}".format(current_move))
        execute_move(current_move, board, pieces)


def execute_move(move_sequence, board, pieces):
    
    for move in move_sequence:

        sort_pieces(pieces, move)

        for i in range(len(pieces)):
            cur_row = pieces[i].movable_row
            cur_col = pieces[i].movable_col

            if (move == "u"):
                newCoords = moveUp(board, cur_row, cur_col)
                pieces[i].movable_row = newCoords[0]

            elif (move == "d"):
                newCoords = moveDown(board, cur_row, cur_col)
                pieces[i].movable_row = newCoords[0]

            elif (move == "l"):
                newCoords = moveLeft(board, cur_row, cur_col)
                pieces[i].movable_col = newCoords[1]

            elif (move == "r"):
                newCoords = moveRight(board, cur_row, cur_col)
                pieces[i].movable_col = newCoords[1]
            
            size_board = int(len(board) ** 0.5)
            board[cur_row * size_board + cur_col] = "."
            board[pieces[i].dest_row * size_board + pieces[i].dest_col] = pieces[i].dest_symbol
            board[newCoords[0] * size_board + newCoords[1]] = pieces[i].movable_symbol
    
    print_board(board)

# --------------------------------------------------------------------------------------------------

def main():
    print("[0] Player")
    print("[1] AI")
    play_choice = input("Game mode: ")

    if (play_choice == "0"):
        player_loop()
        return
    
    # for i in range (0, 2):
    #     lvl = getattr(levels, 'lvl' + str(i))
    (board, pieces) = levels.lvl4()
    print_board(board)
    global nodes_expanded
    nodes_expanded = 0 

    print("Using BFS:")
    # start_mem = memory_usage()[0]
    start = time.time()
    bfs_sol = bfs(board, pieces)
    end = time.time()
    # end_mem = memory_usage()[0]
    bfs_exec_time =  (end - start)*1000
    bfs_nodes = nodes_expanded
    # bfs_mem_usage = (end_mem - start_mem)

    print("Using DFS:")
    # start_mem = memory_usage()[0]
    nodes_expanded = 0 
    start = time.time()
    dfs_sol = dfs(board, pieces)
    end = time.time()
    # end_mem = memory_usage()[0]
    dfs_exec_time =  (end - start)*1000
    dfs_nodes = nodes_expanded
    # dfs_mem_usage = (end_mem - start_mem)

    print("Using Iterative Deepening:")
    # start_mem = memory_usage()[0]
    nodes_expanded = 0 
    start = time.time()
    ids_sol = iterative_deepening(board, pieces)
    end = time.time()
    # end_mem = memory_usage()[0]
    ids_exec_time =  (end - start)*1000
    ids_nodes = nodes_expanded
    # ids_mem_usage = (end_mem - start_mem)

    print("Using A*:")
    # start_mem = memory_usage()[0]
    nodes_expanded = 0 
    start = time.time()
    a_star_sol = a_star(board, pieces)
    end = time.time()
    # end_mem = memory_usage()[0]
    a_star_exec_time =  (end - start)*1000
    a_star_nodes = nodes_expanded
    # a_star_mem_usage = (end_mem - start_mem)

    print_table([("Algorithm",   "No. Moves",        "Sol",        "Exec Time",                 "Nodes Exp",    "Mem Usage"),
                 ("BFS",       str(len(bfs_sol)),   bfs_sol,    str(round(bfs_exec_time)),    str(bfs_nodes),      "N/A"),                 
                 ("DFS",       str(len(dfs_sol)),   dfs_sol,    str(round(dfs_exec_time)),    str(dfs_nodes),      "N/A"),
                 ("IDS",       str(len(ids_sol)),   ids_sol,    str(round(ids_exec_time)),    str(ids_nodes),      "N/A"),
                 ("A*",      str(len(a_star_sol)),  a_star_sol, str(round(a_star_exec_time)), str(a_star_nodes),   "N/A"),
    ])

main()