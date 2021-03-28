import draw

# Given a board and a direction (either 1 or 0 in rowMov and colMov) returns the farthest position in that direction
def getNewPiecePosition(board, curRow, curCol, rowMov, colMov):
    if (rowMov == 0 and colMov == 0): return [curRow, curCol]

    newRow = curRow
    newCol = curCol

    while (True):

        if (newRow + rowMov >= 0 and newRow + rowMov < len(board) and newCol + colMov >= 0 and newCol + colMov < len(board)):

            if (board[newRow + rowMov][newCol + colMov] != "." and board[newRow + rowMov][newCol + colMov] != "P" and board[newRow + rowMov][newCol + colMov] != "T"):
                break # se o mov for para uma casa que não vazia
            else: # else muda para lá a pos atual
                newRow += rowMov
                newCol += colMov
        else:
            break      
    
    # print("Returning: {} {}".format(newRow, newCol))
    return [newRow, newCol]


# -- 4 possible movements --

def moveUp(board, cur_row, cur_col):
    return getNewPiecePosition(board, cur_row, cur_col, -1, 0)

def moveDown(board, cur_row, cur_col):
    return getNewPiecePosition(board, cur_row, cur_col, 1, 0)

def moveLeft(board, cur_row, cur_col):
    return getNewPiecePosition(board, cur_row, cur_col, 0, -1)

def moveRight(board, cur_row, cur_col):
    return getNewPiecePosition(board, cur_row, cur_col, 0, 1)



# Checks if every movable piece has reached its destination
def check_end(pieces):

    for i in range(len(pieces)):
        # accesses tuple on same pos of movable and destination arrays and compares x and y coords
        if (pieces[i].check_coords_inequality()):
            return False

    return True


def sort_pieces(pieces, move):
    if (move == "w"):
        pieces.sort(key=lambda x: x.movable_row, reverse=False)
    elif (move == "s"):
        pieces.sort(key=lambda x: x.movable_row, reverse=True)
    elif (move == "a"):
        pieces.sort(key=lambda x: x.movable_col, reverse=False)
    elif (move == "d"):
        pieces.sort(key=lambda x: x.movable_col, reverse=True)

# Executes the move sequence string one character at a time, updating positions and optionally drawing board after every move
def execute_move_sequence(mutable_board, pieces, move_sequence, draw_move_sequence):

    if (draw_move_sequence):
        print("-- Initial Board --")
        draw.print_board(len(mutable_board), mutable_board)

    for move in move_sequence:

        # sort pieces using coords
        sort_pieces(pieces, move)
        # for example, p . p . = moved to the right should start with the most right "p"
        # moving to the right and only after move the most left "p" to the right 
        for i in range(len(pieces)): 
            
            cur_row = pieces[i].movable_row
            cur_col = pieces[i].movable_col

            if (move == "w"):
                newCoords = moveUp(mutable_board, cur_row, cur_col)
                pieces[i].movable_row = newCoords[0]

            elif (move == "s"):
                newCoords = moveDown(mutable_board, cur_row, cur_col)
                pieces[i].movable_row = newCoords[0]

            elif (move == "a"):
                newCoords = moveLeft(mutable_board, cur_row, cur_col)
                pieces[i].movable_col = newCoords[1]

            elif (move == "d"):
                newCoords = moveRight(mutable_board, cur_row, cur_col)
                pieces[i].movable_col = newCoords[1]
            
            mutable_board[cur_row][cur_col] = "."
            mutable_board[newCoords[0]][newCoords[1]] = pieces[i].movable_symbol
        
        if (draw_move_sequence):
            print("-- After move --")
            draw.print_board(len(mutable_board), mutable_board)
