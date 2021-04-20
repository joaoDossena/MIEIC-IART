from collections import deque
from copy import deepcopy

import globals
import utils
import search_algorithms
import heuristics

# Receives list of pieces.Piece
# Checks if every movable piece has reached its destination
# Returns boolean
def check_end(pieces):
    for i in range(len(pieces)):
        # accesses tuple on same pos of movable and destination arrays and compares x and y coords
        if (pieces[i].check_coords_inequality()):
            return False
    return True


# Receives node state.State
# Calculates its children, here called neighbors
# Returns list of children
def expand(node):
    if(globals.debug):
        print("Expanding node: {} Depth: {}".format(node.move, node.depth))

    globals.nodes_expanded += 1

    neighbours = list()

    if (node.move != ""):
        if (node.move[-1] == "u" or node.move[-1] == "d"):
            if(node.map != move(node,'r').map):
                neighbours.append(move(node, "r"))

            if(node.map != move(node,'l').map):
                neighbours.append(move(node, "l"))
        
        elif (node.move[-1] == "l" or node.move[-1] == "r"):
            if(node.map != move(node,'u').map):
                neighbours.append(move(node, "u"))
            if(node.map != move(node,'d').map):
                neighbours.append(move(node, "d"))
    
    else:
        if(node.map != move(node,'u').map):
            neighbours.append(move(node, "u"))
        if(node.map != move(node,'d').map):
            neighbours.append(move(node, "d"))
        if(node.map != move(node,'r').map):
            neighbours.append(move(node, "r"))
        if(node.map != move(node,'l').map):
            neighbours.append(move(node, "l"))
    
    return neighbours

# Receives node state.State and offset
# Performs adequate move
# Returns new node
def move(node, offset):

    new_node = deepcopy(node)

    sort_pieces(new_node.pieces, offset)

    for i in range(len(new_node.pieces)):
        cur_row = new_node.pieces[i].movable_row
        cur_col = new_node.pieces[i].movable_col

        if (offset == "u"):
            newCoords = moveUp(new_node.board, cur_row, cur_col)
            new_node.pieces[i].movable_row = newCoords[0]

        elif (offset == "d"):
            newCoords = moveDown(new_node.board, cur_row, cur_col)
            new_node.pieces[i].movable_row = newCoords[0]

        elif (offset == "l"):
            newCoords = moveLeft(new_node.board, cur_row, cur_col)
            new_node.pieces[i].movable_col = newCoords[1]

        elif (offset == "r"):
            newCoords = moveRight(new_node.board, cur_row, cur_col)
            new_node.pieces[i].movable_col = newCoords[1]
            
        size_board = int(len(new_node.board) ** 0.5)
        new_node.board[cur_row * size_board + cur_col] = "."
        new_node.board[newCoords[0] * size_board + newCoords[1]] = new_node.pieces[i].movable_symbol

    new_node.calc_map()

    new_node.parent = node
    new_node.move = new_node.parent.move + offset
    new_node.depth = new_node.parent.depth + 1
    return new_node

# Receives list of pieces.Piece and move character
# Sorts list of pieces by the correct order so as to not have movement problems with the order
# Returns nothing
def sort_pieces(pieces, move):
    if (move == "u"):
        pieces.sort(key=lambda x: x.movable_row, reverse=False)
    elif (move == "d"):
        pieces.sort(key=lambda x: x.movable_row, reverse=True)
    elif (move == "l"):
        pieces.sort(key=lambda x: x.movable_col, reverse=False)
    elif (move == "r"):
        pieces.sort(key=lambda x: x.movable_col, reverse=True)

# Receives board, current row and current column of piece
# Returns new position after move
def moveUp(board, cur_row, cur_col):
    return getNewPiecePosition(board, cur_row, cur_col, -1, 0)
def moveDown(board, cur_row, cur_col):
    return getNewPiecePosition(board, cur_row, cur_col, 1, 0)
def moveLeft(board, cur_row, cur_col):
    return getNewPiecePosition(board, cur_row, cur_col, 0, -1)
def moveRight(board, cur_row, cur_col):
    return getNewPiecePosition(board, cur_row, cur_col, 0, 1)


# Receives board, piece current row, piece current column, row move and column move 
# Calculates new position after move
# Returns new position calculated
def getNewPiecePosition(board, curRow, curCol, rowMov, colMov):
    size_board = int(len(board) ** 0.5)

    if (rowMov == 0 and colMov == 0): return [curRow, curCol]

    newRow = curRow
    newCol = curCol

    while (True):
        calc_pos = size_board * (newRow + rowMov) + newCol + colMov

        if (calc_pos >= 0 and calc_pos < len(board) and newRow + rowMov >= 0 and newRow + rowMov < size_board and newCol + colMov >= 0 and newCol + colMov < size_board):
            
            if (board[calc_pos] != "." and board[calc_pos] != "P" and board[calc_pos] != "T"):
                break # if move is to an occupied tile
            else:
                newRow += rowMov
                newCol += colMov
        else:
            break
    
    return [newRow, newCol]

# Receives nothing
# Reads move from human player
# Returns move read
def read_move():
    while(True):
        print("Choose your move:")
        print("up -> u | down -> d | left -> l | right -> r | undo | restart | hint -> h")
        move = input("> ")
        move.lower()

        if(move in ["u", "d", "l", "r", "undo", "restart", "h"]):
            return move

        print("Illegal move!")

# Receives board and list of pieces.Piece
# Allows human player to play until they win
# Returns nothing
def player_loop(board, pieces):
    previous_board = deepcopy(board)
    previous_pieces = deepcopy(pieces)

    original_board = deepcopy(board)
    original_pieces = deepcopy(pieces)

    current_move = ""
    found_first_undo = False

    print()
    utils.print_board(board)

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
            board = deepcopy(original_board)
            pieces = deepcopy(original_pieces)
            current_move = ""
            utils.print_board(board)
            continue

        elif (new_move == "h"):
            hint = search_algorithms.a_star(board, pieces, heuristics.min_string)
            print("Hint: ", hint[0])

        else:
            found_first_undo = False
            previous_board = deepcopy(board)
            previous_pieces = deepcopy(pieces)

            current_move += new_move

        print("Current Move Sequence: {}".format(current_move))
        execute_move(current_move, board, pieces)

# Receives a move sequence, the board, and a list of pieces.Piece
# Executes all moves in the sequence
# Returns nothing
def execute_move(move_sequence, board, pieces):

    if (len(move_sequence) == 0): utils.print_board(board) ; return
    move = move_sequence[-1]

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
        board[newCoords[0] * size_board + newCoords[1]] = pieces[i].movable_symbol
    
    for i in range(len(pieces)):
        if (board[pieces[i].dest_row * size_board + pieces[i].dest_col] == "."):
            board[pieces[i].dest_row * size_board + pieces[i].dest_col] = pieces[i].dest_symbol
    
    utils.print_board(board)
