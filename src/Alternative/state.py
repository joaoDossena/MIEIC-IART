class State:

    def __init__(self, state, parent, move, depth, cost, key, pieces):

        self.state = state

        self.parent = parent

        self.move = move

        self.depth = depth

        self.cost = cost

        self.key = key

        self.pieces = pieces

        if self.state:
            self.map = ''.join(str(e) for e in self.state)
        
    def calc_map(self):
        self.map = ''.join(str(e) for e in self.state)


