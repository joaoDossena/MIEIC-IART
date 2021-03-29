class Piece:

    def __init__(self, symbol, starting_row, starting_col, dest_row, dest_col):
        self.movable_symbol = symbol
        self.movable_row = starting_row
        self.movable_col = starting_col

        self.dest_symbol = symbol.upper()
        self.dest_row = dest_row
        self.dest_col = dest_col

    def __str__(self):
        return "Symbol: " + str(self.movable_symbol) + "\nRow: " + str(self.movable_row) + "\nCol: " + str(self.movable_col)
    
    def check_coords_inequality(self):
        return (self.movable_row != self.dest_row or self.movable_col != self.dest_col)