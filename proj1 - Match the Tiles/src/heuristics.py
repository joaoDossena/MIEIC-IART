# Receives a state.State
# Calculates the euclidean distance between the pieces and adds 1 for the cost of the move
# Returns said calculation
def euclidean_distance(state):
    cost = 1

    for i in range(len(state.pieces)):
        cost += ((state.pieces[i].movable_row - state.pieces[i].dest_row)**2 + (state.pieces[i].movable_col - state.pieces[i].dest_col)**2)**1/2

    return cost

# Receives a state.State
# Calculates the euclidean distance between the pieces, adds 1 for the cost of the move,
# and adds the state depth squared so that it finds the best answer
# Returns said calculation
def min_string(state):
    cost = 1

    for i in range(len(state.pieces)):
        cost += ((state.pieces[i].movable_row - state.pieces[i].dest_row)**2 + (state.pieces[i].movable_col - state.pieces[i].dest_col)**2)**1/2

    cost += state.depth**2

    return cost