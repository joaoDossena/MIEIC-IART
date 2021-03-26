# import other files
import utils
import pieces
import ai
import time

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
    movable = [pieces.movablePiece("p", x, y)]
    destination = [pieces.destinationPiece("P", x1, y1)]
    return (array, movable, destination)


def print_board(size, board):
    for i in range(0,size):
        string = ""
        for k in range (0, size):
                string += "| " + board[i][k] + " "
        print(string + "|")
    return

def read_move():
    while(True):
        move = input("Execute your move: ")
        move.lower()
        if(move in ["w", "a", "s", "d"]):
            return move
        print("Illegal move!")
    

    
def execute_move(move, board, movable):
    for i in range(len(movable)):
        cur_col = movable[i].col
        cur_row = movable[i].row
    
        if (move == "w"):
            newCoords = moveUp(board, cur_row, cur_col)
            movable[i].row = newCoords[0]

        elif (move == "a"):
            newCoords = moveLeft(board, cur_row, cur_col)
            movable[i].col = newCoords[1]
            
        elif (move == "s"):
            newCoords = moveDown(board, cur_row, cur_col)
            movable[i].row = newCoords[0]

        elif (move == "d"):
            newCoords = moveRight(board, cur_row, cur_col)
            movable[i].col = newCoords[1]

        board[cur_row][cur_col] = "."
        board[newCoords[0]][newCoords[1]] = "p"

    return

def valid_move(move, movable, board):
    for i in range(len(movable)):
        cur_col = movable[i].col
        cur_row = movable[i].row
    
        if (move == "w"):
            newCoords = moveUp(board, cur_row, cur_col)
            if (newCoords == [cur_row, cur_col]): 
                return False

        elif (move == "a"):
            newCoords = moveLeft(board, cur_row, cur_col)
            if (newCoords == [cur_row, cur_col]): 
                return False
            
        elif (move == "s"):
            newCoords = moveDown(board, cur_row, cur_col)
            if (newCoords == [cur_row, cur_col]): 
                return False

        elif (move == "d"):
            newCoords = moveRight(board, cur_row, cur_col)
            if (newCoords == [cur_row, cur_col]): 
                return False
        return True


def moveUp(board, cur_row, cur_col):
    return utils.getNewPiecePosition(board, cur_row, cur_col, -1, 0)

def moveDown(board, cur_row, cur_col):
    return utils.getNewPiecePosition(board, cur_row, cur_col, 1, 0)

def moveLeft(board, cur_row, cur_col):
    return utils.getNewPiecePosition(board, cur_row, cur_col, 0, -1)

def moveRight(board, cur_row, cur_col):
    return utils.getNewPiecePosition(board, cur_row, cur_col, 0, 1)


# Checks if every movable piece has reached its destination
def check_end(movable, destination):

    for i in range(len(movable)):
        # accesses tuple on same pos of movable and destination arrays and compares x and y coords
        if (movable[i].row != destination[i].row or movable[i].col != destination[i].col):
            return False
        
    print("Level Completed!\n")
    return True

def game_loop_human(size, board, movable, destination):
    while(True):
        print_board(size, board)
        while(True):
            move = read_move()
            if(valid_move(move, movable, board)):
                execute_move(move, board, movable)
                break
        if(check_end(movable, destination)):
            return

def execute_string_move(string_of_moves, board, movable):
    board_copy = board
    movable_copy = movable
    for c in string_of_moves:
        print(c)
        if(valid_move(c, movable_copy, board_copy)):
            execute_move(c, board_copy, movable_copy)
    return movable_copy

def game_loop_ai(board, bot, movable, destination):

    movable_copy = execute_string_move(bot.get_best_move(), board, movable)
    if(check_end(movable_copy ,destination)):
        return
    
    bot.choose_move(board, movable, destination)

def game_loop_ai2(size, board, movable, destination, bot, last_move):

    while(True):
        time.sleep(0.5)
        print_board(size, board)
        while(True):
            move = bot.choose_move(last_move, board, movable, destination)
            if(valid_move(move, movable, board)):
                execute_move(move, board, movable)
                last_move = move
                break
        if(check_end(movable, destination)):
            return

def main(size):
    (board, movable, destination) = gen_array(size)

    movable = [pieces.movablePiece("p", 1, 0)]
    destination = [pieces.destinationPiece("P", 2, 3)]

    board = [
        [".", ".", ".", "=", "="],
        ["p", ".", ".", ".", "="],
        ["=", ".", ".", "P", "."],
        [".", ".", "=", "=", "."],
        [".", ".", "=", "=", "."],
    ]

    bot = ai.ai(1)
    game_loop_ai(board, bot, movable, destination)
    # game_loop_human(size, board, movable, destination)
    return

main(5)



# x = 2
# y = 4
# x1 = 2
# y1 = 4
# print(check_end([pieces.movablePiece("p", x, y)], [pieces.destinationPiece("P", x1, y1)]))