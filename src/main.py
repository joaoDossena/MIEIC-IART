# import other files
import utils
import pieces
import ai
import draw

import time

from copy import deepcopy

import random

def gen_random_piece(size):
    y = random.randint(0,size-1)
    x = random.randint(0,size-1)
    return (x, y)

def gen_array(size):
    array = []
    for i in range(0, size):
        sub_array = []
        for k in range(0, size):

            if (random.randint(0,size-1) % (size-1) == 0):
                sub_array.append("/")
            else:
                sub_array.append(".")
        array.append(sub_array)
    (x, y) = gen_random_piece(size)
    array[x][y] = "p"
    (x1, y1) = gen_random_piece(size)    
    while(x1 == x and y1 == y):
        (x1, y1) = gen_random_piece(size)
    array[x1][y1] = "P"
    movable = [pieces.Piece("p", x, y, x1, y1)]
    return (array, movable)


def read_move():
    while(True):
        move = input("Execute your move: ")
        move.lower()
        if(move in ["w", "a", "s", "d"]):
            return move
        print("Illegal move!")
    

#     return

# def valid_move(move, movable, board):
#     for i in range(len(movable)):
#         cur_col = movable[i].col
#         cur_row = movable[i].row
    
#         if (move == "w"):
#             newCoords = moveUp(board, cur_row, cur_col)
#             if (newCoords == [cur_row, cur_col]): 
#                 return False

#         elif (move == "a"):
#             newCoords = moveLeft(board, cur_row, cur_col)
#             if (newCoords == [cur_row, cur_col]): 
#                 return False
            
#         elif (move == "s"):
#             newCoords = moveDown(board, cur_row, cur_col)
#             if (newCoords == [cur_row, cur_col]): 
#                 return False

#         elif (move == "d"):
#             newCoords = moveRight(board, cur_row, cur_col)
#             if (newCoords == [cur_row, cur_col]): 
#                 return False
#         return True





# def game_loop_human(size, board, movable, destination):
#     while(True):
#         draw.print_board(size, board)
#         while(True):
#             move = read_move()
#             if(valid_move(move, movable, board)):
#                 execute_move(move, board, movable)
#                 break
#         if(utils.check_end(movable, destination)):
#             return



def game_loop_ai(board, movablePieces, bot):

    # draw.print_board(len(mutable_board), mutable_board)

    while(True):
        mutable_board = deepcopy(board)
        mutable_pieces = deepcopy(movablePieces)

        best_move_sequence = bot.get_best_move()

        draw_move_sequence = True

        # for i in range(len(mutable_pieces)):
        #     print("Row: {} Col: {} \n".format(mutable_pieces[i].row, mutable_pieces[i].col))

        # executes (and optionally draws) move sequence
        utils.execute_move_sequence(mutable_board, mutable_pieces, best_move_sequence, draw_move_sequence)

        # for i in range(len(mutable_pieces)):
        #     print("Row: {} Col: {} \n".format(mutable_pieces[i].row, mutable_pieces[i].col))

        # checks if move results in solution
        if (utils.check_end(mutable_pieces)):
            print("FOUND SOLUTION")
            print("Move Sequence Found: {}".format(best_move_sequence))
            print("Number of Moves: {}".format(len(best_move_sequence)))
            break

        # consume from queue current best move and add possible moves from it to the heap queue
        bot.choose_move(mutable_board, mutable_pieces)

        
        print(bot.get_move_queue())
        print("\n")
        


def main(size):
    # (board, movablePieces, destinationTiles) = gen_array(size)

    movablePieces = [pieces.Piece("p", 1, 0, 4, 0), pieces.Piece("t", 1, 2, 2, 1)]

    board = [
        [".", ".", ".", "=", "="],
        ["p", ".", "t", ".", "="],
        ["=", "T", ".", ".", "."],
        [".", ".", "=", "=", "."],
        ["P", ".", "=", "=", "."],
    ]

    bot = ai.ai(1)
    game_loop_ai(board, movablePieces, bot)

    # game_loop_ai(board, bot, movable, destination)
    # game_loop_human(size, board, movable, destination)
    return

main(5)



# x = 2
# y = 4
# x1 = 2
# y1 = 4
# print(utils.check_end([pieces.movablePiece("p", x, y)], [pieces.destinationPiece("P", x1, y1)]))