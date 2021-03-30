from collections import deque
from state import State
from pieces import Piece
from copy import deepcopy
import itertools
import levels
from heapq import heappush, heappop, heapify

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


# Checks if every movable piece has reached its destination
def check_end(pieces):

    for i in range(len(pieces)):
        # accesses tuple on same pos of movable and destination arrays and compares x and y coords
        if (pieces[i].check_coords_inequality()):
            return False

    return True


def bfs(start_state, pieces):

    global goal_node, max_search_depth

    # explored, queue = set(), deque([State(start_state, None, "", 0, 0, 0, pieces)])
    
    explored, queue = set(), deque([State(start_state, None, "", 0, 0, 0, pieces)])

    # print("Starting Board")
    # print_board(start_state)
    # print()

    while queue:

        node = queue.popleft()

        # if (node.move == "r"):
        #     print("r")
        #     for piece in node.pieces: print(piece)
        #     print_board(node.state)
        #     print()
        # if (node.move == "ru"):
        #     print("ru")
        #     print_board(node.state)
        #     print()
        # if (node.move == "rul"):
        #     print("rul")
        #     print_board(node.state)
        #     print()
        # if (node.move == "rulu"):
        #     print("rulu")
        #     print_board(node.state)
        #     print()
        # if (node.move == "rulur"):
        #     print("rulur")
        #     print_board(node.state)
        #     print()
        # if (node.move == "rulurd"):
        #     print("rulurd")
        #     print_board(node.state)
        #     print()
        # if (node.move == "rulurdl"):
        #     print("rulurdl")
        #     print_board(node.state)
        #     print()
        # if (node.move == "rulurdlu"):
        #     print("rulurdlu")
        #     print_board(node.state)
        #     print()

        # print_board(node.state)
        # for piece in node.pieces:
        #     print(piece)

        explored.add(node.map)

        if (check_end(node.pieces)):
            print("Solution: {}".format(node.move))
            print_board(node.state)
            break

        neighbours = expand(node)

        for neighbour in neighbours:

            # print_board(neighbour.state)

            if neighbour.map not in explored:
                # print("Adding to Queue")
                queue.append(neighbour)
                if neighbour.depth > max_search_depth:
                    max_search_depth += 1

def dfs(start_state, pieces):

    global max_frontier_size, goal_node, max_search_depth

    explored, stack = set(), list([State(start_state, None, "", 0, 0, 0, pieces)])

    while stack:

        node = stack.pop()

        explored.add(node.map)

        # if node.state == goal_state:
        #     goal_node = node
        #     return stack

        if (check_end(node.pieces)):
            print("Solution: {}".format(node.move))
            print_board(node.state)
            break

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
        # print("movable row: {} col: {} dest row: {} col: {}".format(movable[i].row, movable[i].col, destination[i].row, destination[i].col))
        # print("Cost ", cost)
        
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

        # if node[2].state == goal_state:
        #     goal_node = node[2]
        #     return heap

        if (check_end(node[2].pieces)):
            print("Solution: {}".format(node[2].move))
            print_board(node[2].state)
            break

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

            # if node.state == goal_state:
            #     goal_node = node
            #     return stack

            # print(node.move)

            # if (node.move == "rulurdlu"):
            #     print_board(node.state)
            #     return 

            if (check_end(node.pieces)):
                print("Solution: {}".format(node.move))
                print_board(node.state)
                return
            

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

    # print_board(node.state)
    # neighbours.append(State(move(node, "u"), node, "u", node.depth + 1, node.cost + 1, 0))
    
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

    # print("New Node Move: {}".format(new_node.move))
    new_node.calc_map()

    new_node.parent = node
    new_node.move = new_node.parent.move + offset
    new_node.depth = new_node.parent.depth + 1
    # print("Map: {}".format(new_node.map))
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
        print("up -> u  down -> d  left -> l  right -> r")
        move = input("> ")
        move.lower()

        if(move in ["u", "d", "l", "r", "undo"]):
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
        # print("beginning of while - cur move: {}".format(len(current_move)))

        if (check_end(pieces)):
            print("Level Completed!")
            break

        print()
        new_move = read_move()

        if (new_move == "undo" and found_first_undo == False):
            found_first_undo = True
            # print(len(current_move))
            # undo_move(current_move, board, pieces)
            board = previous_board
            pieces = previous_pieces
            current_move = current_move[:-1]

        elif (new_move == "undo"):
            print("Second Undo in a Row. not allowed :p")
            pass
            
        # elif (new_move == "restart"):
        #     found_first_undo = False
        #     board = original_board
        #     pieces = original_pieces
        #     current_move = ""

        else:
            found_first_undo = False
            previous_board = deepcopy(board)
            previous_pieces = deepcopy(pieces)

            current_move += new_move

        print("Current Move Sequence: {}".format(current_move))
        execute_move(current_move, board, pieces)


def execute_move(move_sequence, board, pieces):

    # mutable_board = deepcopy(board)
    # mutable_pieces = deepcopy(pieces)
    
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


# -----------------------------------------

def main():

    print("[0] Player")
    print("[1] AI")
    play_choice = input("Game mode: ")

    if (play_choice == "0"):
        player_loop()
        return
    
    

    # for i in range (0, 2):
    #     lvl = getattr(levels, 'lvl' + str(i))
    # (board, pieces) = levels.lvl1()
    # print_board(board)
    # print("Using BFS:")
    # bfs(board, pieces)

    # print("Using DFS:")
    # dfs(board, pieces)

        # print("Using Iterative Deepening:")
        #iterative_deepening(board, pieces)
        # print("Using A*:")
        # a_star(board, pieces)

    # curRow = pieces[0].movable_row
    # curCol = pieces[0].movable_col
    # rowMov = 0
    # colMov = 1
        
    # returned_array = getNewPiecePosition(board, curRow, curCol, rowMov, colMov)

    # print("Returned: [{}, {}]".format(returned_array[0], returned_array[1]))




main()