import utils
import time

def print_board(size, board):
    for i in range(0,size):
        string = ""
        for k in range (0, size):
                string += "| " + board[i][k] + " "
        print(string + "|")
    return


def draw_move_sequence(mutable_board, movablePieces, move_sequence):

    print("-- Initial Board --")
    print_board(len(mutable_board), mutable_board)

    for move in move_sequence:

        time.sleep(1)

        # TODO need to sort pieces using coords
        # for example, p . p . = moved to the right should start with the most right "p"
        # moving to the right and only after move the most left "p" to the right 
        for i in range(len(movablePieces)): 
            
            cur_row = movablePieces[i].row
            cur_col = movablePieces[i].col

            if (move == "w"):
                newCoords = utils.moveUp(mutable_board, cur_row, cur_col)
                movablePieces[i].row = newCoords[0]

            elif (move == "s"):
                newCoords = utils.moveDown(mutable_board, cur_row, cur_col)
                movablePieces[i].row = newCoords[0]

            elif (move == "a"):
                newCoords = utils.moveLeft(mutable_board, cur_row, cur_col)
                movablePieces[i].col = newCoords[1]

            elif (move == "d"):
                newCoords = utils.moveRight(mutable_board, cur_row, cur_col)
                movablePieces[i].col = newCoords[1]
            
            mutable_board[cur_row][cur_col] = "."
            mutable_board[newCoords[0]][newCoords[1]] = "p"
        
        print("-- After move --")
        print_board(len(mutable_board), mutable_board)


