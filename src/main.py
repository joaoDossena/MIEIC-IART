# import other files
import utils
import movablePiece

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
            sub_array.append(".")
        array.append(sub_array)
    (x, y) = gen_random_piece(size)
    array[x][y] = "p"
    (x1, y1) = gen_random_piece(size)    
    while(x1 == x and y1 == y):
        (x1, y1) = gen_random_piece(size)
    array[x1][y1] = "P"


    movable = [movablePiece.movablePiece("p", x, y)]

    # print(movable[0].col)

    destination = [movablePiece.destinationPiece("P", x1, y1)]

    # print(destination[0].col)
    
    return (array, movable, destination)


def print_board(size, board):
    for i in range(0,size):
        string = ""
        for k in range (0, size):
                string += "| " + board[i][k] + " "
        print(string + "|")
    return

def read_move():
    move = input("Execute your move: ")
    move.lower()
    if(move not in ["w", "a", "s", "d"]):
        print("Illegal move!")
        return
    return move

def execute_move(move, size, board, movable):

    for i in range(len(movable)):
        cur_col = movable[i].col
        cur_row = movable[i].row
    
        if (move == "w"):
            newCoords = moveUp(board, cur_row, cur_col)

            if (len(newCoords) == 0): continue
            else: movable[i].row = newCoords[0]

        elif (move == "a"):
            newCoords = moveLeft(board, cur_row, cur_col)

            if (len(newCoords) == 0): continue
            else: movable[i].col = newCoords[1]
            
        elif (move == "s"):
            newCoords = moveDown(board, cur_row, cur_col)

            if (len(newCoords) == 0): continue
            else: movable[i].row = newCoords[0]

        elif (move == "d"):
            newCoords = moveRight(board, cur_row, cur_col)

            if (len(newCoords) == 0): continue
            else: movable[i].col = newCoords[1]

        board[cur_row][cur_col] = "."
        board[newCoords[0]][newCoords[1]] = "p"

    return


def moveUp(board, cur_row, cur_col):
    # Get the most "up" position (max "slide")
    [newRow, newCol] = utils.getNewPiecePosition(board, cur_row, cur_col, -1, 0)
    if (newRow == cur_row and newCol == cur_col): return []
    print("\nMoving Up\n")
    return [newRow, newCol]
       

def moveDown(board, cur_row, cur_col):

    [newRow, newCol] = utils.getNewPiecePosition(board, cur_row, cur_col, 1, 0)
    if (newRow == cur_row and newCol == cur_col): return []
    print("\nMoving Down\n")      
    return [newRow, newCol]


def moveLeft(board, cur_row, cur_col):

    [newRow, newCol] = utils.getNewPiecePosition(board, cur_row, cur_col, 0, -1)
    if (newRow == cur_row and newCol == cur_col): return []
    print("\nMoving Left\n")   
    return [newRow, newCol]


def moveRight(board, cur_row, cur_col):

    [newRow, newCol] = utils.getNewPiecePosition(board, cur_row, cur_col, 0, 1)
    if (newRow == cur_row and newCol == cur_col): return []
    print("\nMoving Right\n")       
    return [newRow, newCol]



# btw... tá a funcionar
# Checks if every movable piece has reached its destination
def check_end(movable, destination):

    for i in range(len(movable)):
        # accesses tuple on same pos of movable and destination arrays and compares x and y coords
        if (movable[i].row != destination[i].row or movable[i].col != destination[i].col):
            return False
    return True

def game_loop(size, board, movable, destination):
    while(True):
        print_board(size, board)
        move = read_move()
        execute_move(move, size, board, movable)
        if(check_end(movable, destination)):
            return

def main(size):
    (board, movable, destination) = gen_array(size)
    game_loop(size, board, movable, destination)
    return

main(5)



# x = 2
# y = 4
# x1 = 2
# y1 = 4
# print(check_end([movablePiece.movablePiece("p", x, y)], [movablePiece.destinationPiece("P", x1, y1)]))