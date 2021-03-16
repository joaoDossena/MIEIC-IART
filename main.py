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
    
    return array


def print_board(size, board):
    for i in range(0,size):
        string = ""
        for k in range (0, size):
                string += "| " + board[i][k] + " "
        print(string + "|")
    return

def read_move():
    move = input("Execute your move: ")
    if(move not in ["w", "a", "s", "d", "W", "A", "S", "D"]):
        print("Illegal move!")
        return
    execute_move(move)
    return

def execute_move(move):
    return

def check_end(board):
    return

def game_loop(size, board):
    while(True):
        print_board(size, board)
        read_move()
        execute_move()
        if(check_end()):
            return

def main(size):
    board = gen_array(size)
    game_loop(size, board)
    return

main(5)