# import other files
import utils

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

    movable = [("p", x, y)]

    destination = [("P", x1, y1)]
    
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
    
    if (move == "w"): moveUp(board, movable)
    elif (move == "a"): moveLeft(board, movable)
    elif (move == "s"): moveDown(board, movable)
    elif (move == "d"): moveRight(board, movable)

    return


# Get the most left/right/down/up position (max "slide")



def moveUp(board, movable):

    for i in range(len(movable)):
        cur_col = movable[i][2]
        cur_row = movable[i][1]

        [newRow, newCol] = utils.getNewPiecePosition(board, cur_row, cur_col, -1, 0)
        
        if (newRow == cur_row and newCol == cur_col): return
       
        print("\nMoving Up\n")
    
        board[cur_row][cur_col] = "."
        board[newRow][newCol] = "p"
            
        lst = list(movable[i])
        lst[1] = newRow
        movable[i] = tuple(lst)
        # print(movable)

def moveDown(board, movable):

    for i in range(len(movable)):
        cur_col = movable[i][2]
        cur_row = movable[i][1]

        [newRow, newCol] = utils.getNewPiecePosition(board, cur_row, cur_col, 1, 0)

        if (newRow == cur_row and newCol == cur_col): return

        print("\nMoving Down\n")
            
        board[cur_row][cur_col] = "."
        board[newRow][newCol] = "p"
            
        lst = list(movable[i])
        lst[1] = newRow
        movable[i] = tuple(lst)
            



def moveLeft(board, movable):

    for i in range(len(movable)):
        cur_col = movable[i][2]
        cur_row = movable[i][1]

        [newRow, newCol] = utils.getNewPiecePosition(board, cur_row, cur_col, 0, -1)

        if (newRow == cur_row and newCol == cur_col): return

        print("\nMoving Left\n")
            
        board[cur_row][cur_col] = "."
        board[newRow][newCol] = "p"
            
        lst = list(movable[i])
        lst[2] = newCol
        movable[i] = tuple(lst)
            


def moveRight(board, movable):

    for i in range(len(movable)):
        cur_col = movable[i][2]
        cur_row = movable[i][1]

        [newRow, newCol] = utils.getNewPiecePosition(board, cur_row, cur_col, 0, 1)

        if (newRow == cur_row and newCol == cur_col): return

        print("\nMoving Right\n")
            
        board[cur_row][cur_col] = "."
        board[newRow][newCol] = "p"
            
        lst = list(movable[i])
        lst[2] = newCol
        movable[i] = tuple(lst)

    



# btw... tá a funcionar
# Checks if every movable piece has reached its destination
def check_end(movable, destination):

    for i in range(len(movable)):
        # accesses tuple on same pos of movable and destination arrays and compares x and y coords
        if (movable[i][1] != destination[i][1] or movable[i][2] != destination[i][2]):
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
# print(check_end([("p", x, y)], [("P", x1, y1)]))