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

def lvl1(): # Level 19 -> 8 moves for perfect score
    board = [
        ".", "T", "=", ".", 
        "p", ".", "P", ".",
        ".", ".", ".", ".",
        "=", ".", "t", ".",
    ]
    pcs = [pieces.Piece("p", 1, 0, 1, 2), pieces.Piece("t", 3, 2, 0, 1)]
    return (board, pcs)

def lvl2(): # Level 20 -> 8 moves for perfect score
    board = [
        ".", "=", "=", ".", 
        "t", ".", ".", ".",
        "P", "=", "=", "p",
        "T", ".", ".", ".",
    ]
    pcs = [pieces.Piece("p", 2, 3, 2, 0), pieces.Piece("t", 1, 0, 3, 0)]
    return (board, pcs)

def lvl3(): # Level 21 -> 9 moves for perfect score
    board = [
        "T", ".", "p", "t", 
        "=", ".", "=", "P",
        "=", "=", ".", ".",
        ".", ".", "=", "=",
    ]
    pcs = [pieces.Piece("p", 0, 2, 1, 3), pieces.Piece("t", 0, 3, 0, 0)]
    return (board, pcs)

def lvl4(): # Level 22 -> 9 moves for perfect score
    board = [
        ".", "p", ".", "=", 
        "=", "t", ".", "T",
        "=", ".", "=", "P",
        "=", ".", "=", "=",
    ]
    pcs = [pieces.Piece("p", 0, 1, 2, 3), pieces.Piece("t", 1, 1, 1, 3)]
    return (board, pcs)

