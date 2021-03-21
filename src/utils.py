# Given a board and a direction (either 1 or 0 in rowMov and colMov) returns the farthest position in that direction
def getNewPiecePosition(board, curRow, curCol, rowMov, colMov):
    if (rowMov == 0 and colMov == 0): return [curRow, curCol]

    newRow = curRow
    newCol = curCol

    while (True):

        if (newRow + rowMov >= 0 and newRow + rowMov < len(board) and newCol + colMov >= 0 and newCol + colMov < len(board)):

            if (board[newRow + rowMov][newCol + colMov] != "." and board[newRow + rowMov][newCol + colMov] != "P" and board[newRow + rowMov][newCol + colMov] != "p"):
                break # se o mov for para uma casa que não vazia
            else: # else muda para lá a pos atual
                newRow += rowMov
                newCol += colMov
        else:
            break      
    
    # print("Returning: {} {}".format(newRow, newCol))
    return [newRow, newCol]


# board = [
#     [".", "/", "."],
#     [".", ".", "."],
#     [".", ".", "."]
# ]

# startRow = 2
# startCol = 0

# [endRow, endCol] = getNewPiecePosition(board, startRow, startCol, 0, 0)
# print("Start Position: {} {}".format(startRow, startCol))
# print("End   Position: {} {}".format(endRow, endCol))