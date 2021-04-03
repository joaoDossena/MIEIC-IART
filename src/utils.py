# Receives 2D matrix
# Prints matrix as a nice board
# Returns nothing
def print_board(board):
    side_len = int(len(board) ** 0.5)
    print( (side_len * 2 + 3) * "-")
    for i in range(side_len):
        print("| ", end="")
        for j in range(side_len):
            print(board[i * side_len + j] + " ", end="")
        print("|")
    print( (side_len * 2 + 3) * "-")

# Receives list of tuples of strings
# Prints said argument as a nice table
# Returns nothing
def print_table(table):
    col_width = [max(len(x) for x in col) for col in zip(*table)]
    for line in table:
        print ("| " + " | ".join("{:{}}".format(x, col_width[i])
                                for i, x in enumerate(line)) + " |")