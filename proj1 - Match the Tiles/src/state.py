class State:

    def __init__(self, board, parent, move, depth, cost, key, pieces):

        self.board = board

        self.parent = parent

        self.move = move

        self.depth = depth

        self.cost = cost

        self.key = key

        self.pieces = pieces

        if self.board:
            self.map = ''.join(str(e) for e in self.board)
    
    def __str__(self):
        return ''.join(str(e) for e in self.board) + " " + str(len(self.pieces))


    def calc_map(self):
        self.map = ''.join(str(e) for e in self.board)


