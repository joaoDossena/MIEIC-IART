from collections import deque
from state import State
from pieces import Piece
from copy import deepcopy
import itertools
from heapq import heappush, heappop, heapify

initial_state = list()

nodes_expanded = 0
max_search_depth = 0


def print_board(board):
    side_len = int(len(board) ** 0.5)
    for i in range(side_len):
        for j in range(side_len):
            print(board[i * side_len + j] + " ", end="")
        print()



def bfs(start_state, pieces):

    global goal_node, max_search_depth

    # explored, queue = set(), deque([State(start_state, None, "", 0, 0, 0, pieces)])
    
    explored, queue = set(), deque([State(start_state, None, "", 0, 0, 0, pieces)])

    while queue:

        node = queue.popleft()

        explored.add(node.map)

        # if (node.move == "u"):
        # if (node.move == "ur"):
        # if (node.move == "urd"):
        if (node.move == "urdl"):
            print_board(node.state)
            break

        neighbours = expand(node)

        for neighbour in neighbours:

            # print_board(neighbour.state)

            if neighbour.map not in explored:
                # print("Adding to Queue")
                queue.append(neighbour)
                # if neighbour.depth > max_search_depth:
                #     max_search_depth += 1

def dfs(start_state, pieces):

    global max_frontier_size, goal_node, max_search_depth

    explored, stack = set(), list([State(start_state, None, "", 0, 0, 0, pieces)])

    while stack:

        node = stack.pop()

        explored.add(node.map)

        # if node.state == goal_state:
        #     goal_node = node
        #     return stack

        if (node.move == "urdl"):
            print_board(node.state)
            break

        neighbors = reversed(expand(node))

        for neighbor in neighbors:
            if neighbor.map not in explored:
                stack.append(neighbor)
                explored.add(neighbor.map)

        #         if neighbor.depth > max_search_depth:
        #             max_search_depth += 1

        # if len(stack) > max_frontier_size:
        #     max_frontier_size = len(stack)

# def h(state):
#     cost = 1
#     for i in range(len(state.pieces)):
#             cost += ((state.pieces[i].movable_row - state.pieces[i].dest_row)**2 + (state.pieces[i].movable_col - state.pieces[i].dest_col)**2)**1/2
#             # print("movable row: {} col: {} dest row: {} col: {}".format(movable[i].row, movable[i].col, destination[i].row, destination[i].col))

#             # print(cost)
        
#     return cost

# def a_star(start_state, pieces):

#     global max_frontier_size, goal_node, max_search_depth

#     explored, heap, heap_entry, counter = set(), list(), {}, itertools.count()

#     key = h(start_state)

#     root = State(start_state, None, "", 0, 0, key, pieces)

#     entry = (key, 0, root)

#     heappush(heap, entry)

#     heap_entry[root.map] = entry

#     while heap:

#         node = heappop(heap)

#         explored.add(node[2].map)

#         # if node[2].state == goal_state:
#         #     goal_node = node[2]
#         #     return heap

#         if (node[2].move == "urdl"):
#             print_board(node.state)
#             break

#         neighbors = expand(node[2])

#         for neighbor in neighbors:

#             neighbor.key = neighbor.cost + h(neighbor.state)

#             entry = (neighbor.key, neighbor.move, neighbor)

#             if neighbor.map not in explored:

#                 heappush(heap, entry)

#                 explored.add(neighbor.map)

#                 heap_entry[neighbor.map] = entry

#                 if neighbor.depth > max_search_depth:
#                     max_search_depth += 1

#             elif neighbor.map in heap_entry and neighbor.key < heap_entry[neighbor.map][2].key:

#                 hindex = heap.index((heap_entry[neighbor.map][2].key,
#                                      heap_entry[neighbor.map][2].move,
#                                      heap_entry[neighbor.map][2]))

#                 heap[int(hindex)] = entry

#                 heap_entry[neighbor.map] = entry

#                 heapify(heap)

#         if len(heap) > max_frontier_size:
#             max_frontier_size = len(heap)



def iterative_deepening(start_state, pieces):
    global max_frontier_size, goal_node, max_search_depth

    explored, stack = set(), list([State(start_state, None, "", 0, 0, 0, pieces)])

    while stack:

        node = stack.pop()

        explored.add(node.map)

        # if node.state == goal_state:
        #     goal_node = node
        #     return stack

        if (node.move == "urdl"):
            print_board(node.state)
            break

        neighbors = reversed(expand(node))

        for neighbor in neighbors:
            if neighbor.map not in explored:
                stack.append(neighbor)
                explored.add(neighbor.map)

                if neighbor.depth > max_search_depth:
                    max_search_depth += 1

        # if len(stack) > max_frontier_size:
        #     max_frontier_size = len(stack)


def expand(node):

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
            new_node.parent = node
            new_node.move = new_node.parent.move + "u"
            newCoords = moveUp(new_node.state, cur_row, cur_col)
            new_node.pieces[i].movable_row = newCoords[0]

        elif (offset == "d"):
            new_node.parent = node
            new_node.move = new_node.parent.move + "d"
            newCoords = moveDown(new_node.state, cur_row, cur_col)
            new_node.pieces[i].movable_row = newCoords[0]

        elif (offset == "l"):
            new_node.parent = node
            new_node.move = new_node.parent.move + "l"
            newCoords = moveLeft(new_node.state, cur_row, cur_col)
            new_node.pieces[i].movable_col = newCoords[1]

        elif (offset == "r"):
            new_node.parent = node
            new_node.move = new_node.parent.move + "r"
            newCoords = moveRight(new_node.state, cur_row, cur_col)
            new_node.pieces[i].movable_col = newCoords[1]
            
        size_board = int(len(new_node.state) ** 0.5)
        new_node.state[cur_row * size_board + cur_col] = "."
        new_node.state[newCoords[0] * size_board + newCoords[1]] = new_node.pieces[i].movable_symbol

    print("New Node Move: {}".format(new_node.move))
    new_node.calc_map()

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

        if (calc_pos >= 0 and calc_pos < len(board) and newRow + rowMov >= 0 and newCol + colMov >= 0):
            
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

def main():

    board = [
        ".", ".", ".", "=", "=",
        "p", ".", "t", ".", "=",
        "=", "T", ".", ".", ".",
        ".", ".", "=", "=", ".",
        "P", ".", "=", "=", ".",
    ]

        # ...==
        # ....=
        # =t...
        # ..==p
        # P.==.

    pieces = [Piece("p", 1, 0, 4, 0), Piece("t", 1, 2, 2, 1)]

    # row = 1
    # col = 0
    # print(getNewPiecePosition(board, row, col, 1, 0))

    # bfs(board, pieces)
    # dfs(board, pieces)
    # a_star(board, pieces)
    iterative_deepening(board, pieces)


    # boardddd = [
    #     ".", ".", ".", "=", "=",
    #     ".", ".", ".", ".", "=",
    #     "=", "T", "t", ".", ".",
    #     ".", ".", "=", "=", ".",
    #     "P", "p", "=", "=", ".",
    # ]

    # print(getNewPiecePosition(boardddd, 4, 0, 0, -1))

    # print(''.join(str(e) for e in goal_board))



main()