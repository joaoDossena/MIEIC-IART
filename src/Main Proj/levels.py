import pieces

# TEMPLATE 4x4
# board = [
#         ".", ".", ".", ".", 
#         ".", ".", ".", ".",
#         ".", ".", ".", ".",
#         ".", ".", ".", ".",
#     ]

def lvl0():
    board = [
        ".", ".", ".", "=", "=",
        "p", ".", "t", ".", "=",
        "=", "T", ".", ".", ".",
        ".", ".", "=", "=", ".",
        "P", ".", "=", "=", ".",
    ]
    pcs = [pieces.Piece("p", 1, 0, 4, 0), pieces.Piece("t", 1, 2, 2, 1)]
    return (board, pcs)

def lvl1(): # Level 19 - 8 moves for perfect score
    board = [
        ".", "T", "=", ".", 
        "p", ".", "P", ".",
        ".", ".", ".", ".",
        "=", ".", "t", ".",
    ]
    pcs = [pieces.Piece("p", 1, 0, 1, 2), pieces.Piece("t", 3, 2, 0, 1)]
    return (board, pcs)

def lvl2():
    board = []
    pieces = []
    return (board, pieces)

def lvl3():
    board = []
    pieces = []
    return (board, pieces)

    