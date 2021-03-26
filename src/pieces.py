class movablePiece:

    def __init__(self, symbol, startingRow, startingCol):
        self.symbol = symbol
        self.row = startingRow
        self.col = startingCol

    def updatePos(self, newRow, newCol):
        self.row = newRow
        self.col = newCol


# # Testing
# p = movablePiece(".", 1, 2)
# print("{} {}".format(p.row, p.col))
# p.updatePos(5, 4)
# print("{} {}".format(p.row, p.col))

class destinationPiece:

    def __init__(self, symbol, row, col):
        self.symbol = symbol
        self.row = row
        self.col = col