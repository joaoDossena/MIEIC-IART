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
        up_row = cur_row - 1
        # print('{} {}'.format(cur_row, cur_col))

        if (up_row >= 0 and "." == board[up_row][movable[i][2]]):
            print("\nMoving Up\n")
    
            board[cur_row][cur_col] = "."
            board[up_row][cur_col] = "p"
            
            lst = list(movable[i])
            lst[1] = up_row
            movable[i] = tuple(lst)
            # print(movable)

def moveDown(board, movable):

    for i in range(len(movable)):
        cur_col = movable[i][2]
        cur_row = movable[i][1]
        down_row = cur_row + 1

        if (down_row < len(board) and "." == board[down_row][movable[i][2]]):
            print("\nMoving Down\n")
            
            board[cur_row][cur_col] = "."
            board[down_row][cur_col] = "p"
            
            lst = list(movable[i])
            lst[1] = down_row
            movable[i] = tuple(lst)
            



def moveLeft(board, movable):

    for i in range(len(movable)):
        cur_col = movable[i][2]
        cur_row = movable[i][1]
        left_col = cur_col - 1

        if (left_col >= 0 and "." == board[movable[i][1]][left_col]):
            print("\nMoving Left\n")

            board[cur_row][cur_col] = "."
            board[cur_row][left_col] = "p"


            lst = list(movable[i])
            lst[2] = left_col
            movable[i] = tuple(lst)
            


def moveRight(board, movable):

    for i in range(len(movable)):
        cur_col = movable[i][2]
        cur_row = movable[i][1]
        right_col = cur_col + 1

        if (right_col < len(board) and "." == board[movable[i][1]][right_col]):
            print("\nMoving Right\n")

            board[cur_row][cur_col] = "."
            board[cur_row][right_col] = "p"

            lst = list(movable[i])
            lst[2] = right_col
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