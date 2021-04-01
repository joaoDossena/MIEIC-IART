import pieces

# TEMPLATE 4x4
# board = [
#         ".", ".", ".", ".", 
#         ".", ".", ".", ".",
#         ".", ".", ".", ".",
#         ".", ".", ".", ".",
#     ]

def lvl1(): 
    board = [
        "P", ".", ".", "p", 
        "=", ".", ".", ".",
        ".", ".", ".", "=",
        "T", ".", "t", ".",
    ]
    pcs = [pieces.Piece("p", 0, 3, 0, 0), pieces.Piece("t", 3, 2, 3, 0)]
    return (board, pcs)

def lvl2(): 
    board = [
        ".", ".", "=", "P", 
        ".", "=", "p", "T",
        ".", ".", ".", ".",
        ".", ".", "=", "t",
    ]
    pcs = [pieces.Piece("p", 1, 2, 0, 3), pieces.Piece("t", 3, 3, 1, 3)]
    return (board, pcs)

def lvl3(): 
    board = [
        ".", ".", ".", "P", 
        ".", ".", "T", "=",
        "p", ".", ".", "=",
        ".", ".", ".", "t",
    ]
    pcs = [pieces.Piece("p", 2, 0, 0, 3), pieces.Piece("t", 3, 3, 1, 2)]
    return (board, pcs)

def lvl4(): 
    board = [
        "P", ".", "p", "=", 
        "=", "=", ".", "=",
        "=", ".", ".", ".",
        "t", ".", "T", "=",
    ]
    pcs = [pieces.Piece("p", 0, 2, 0, 0), pieces.Piece("t", 3, 0, 3, 2)]
    return (board, pcs)

def lvl5(): 
    board = [
        "=", "t", "=", "P", 
        "=", "p", "T", ".",
        "=", ".", ".", "=",
        ".", "=", "=", ".",
    ]
    pcs = [pieces.Piece("p", 1, 1, 0, 3), pieces.Piece("t", 0, 1, 1, 2)]
    return (board, pcs)

def lvl6(): 
    board = [
        "=", ".", ".", "T", 
        "=", ".", ".", "=",
        ".", "p", ".", "P",
        ".", ".", "=", "t",
    ]
    pcs = [pieces.Piece("t", 3, 3, 0, 3), pieces.Piece("p", 2, 1, 2, 3)]
    return (board, pcs)

def lvl7(): 
    board = [
        "T", ".", "=", "=", 
        "P", ".", "t", "p",
        ".", "=", "=", "=",
        ".", "=", ".", "=",
    ]
    pcs = [pieces.Piece("p", 1, 3, 1, 0), pieces.Piece("t", 1, 2, 0, 0)]
    return (board, pcs)

def lvl8(): 
    board = [
        "=", ".", ".", ".", 
        "p", ".", ".", ".",
        "t", ".", ".", "=",
        ".", ".", "T", "P",
    ]
    pcs = [pieces.Piece("p", 1, 0, 3, 3), pieces.Piece("t", 2, 0, 3, 2)]
    return (board, pcs)

def lvl9():
    board = [
        ".", "=", "p", ".", 
        ".", ".", ".", ".",
        ".", ".", ".", "T",
        ".", "t", "P", "=",
    ]
    pcs = [pieces.Piece("p", 0, 2, 3, 2), pieces.Piece("t", 3, 1, 2, 3)]
    return (board, pcs)

def lvl10():
    board = [
        ".", ".", "P", "=", 
        "=", ".", "t", ".",
        "p", ".", ".", "T",
        "=", "=", "=", "=",
    ]
    pcs = [pieces.Piece("p", 2, 0, 0, 2), pieces.Piece("t", 1, 2, 2, 3)]
    return (board, pcs)
    
def lvl11():
    board = [
        "=", "P", "=", "T", 
        ".", "t", ".", ".",
        ".", ".", "=", "p",
        "=", ".", "=", "=",
    ]
    pcs = [pieces.Piece("p", 2, 3, 0, 1), pieces.Piece("t", 1, 1, 0, 3)]
    return (board, pcs)


def lvl12():
    board = [
        "=", ".", "T", "=", 
        ".", "=", "t", ".",
        "=", "p", ".", ".",
        "=", "=", "P", "=",
    ]
    pcs = [pieces.Piece("p", 2, 1, 3, 2), pieces.Piece("t", 1, 2, 0, 2)]
    return (board, pcs)
##################################################################################
def lvl13(): 
    board = [
        ".", "=", "P", "T", 
        "=", ".", ".", "=",
        "=", ".", "p", ".",
        "t", ".", ".", ".",
    ]
    pcs = [pieces.Piece("p", 2, 2, 0, 2), pieces.Piece("t", 3, 0, 0, 3)]
    return (board, pcs)

def lvl14(): 
    board = [
        "T", "=", "=", "=", 
        ".", "t", "=", "=",
        ".", ".", ".", "=",
        "p", "=", ".", "P",
    ]
    pcs = [pieces.Piece("p", 3, 0, 3, 3), pieces.Piece("t", 1, 1, 0, 0)]
    return (board, pcs)

def lvl15(): 
    board = [
        "t", ".", ".", ".", 
        ".", ".", ".", "p",
        "P", ".", ".", ".",
        ".", "=", "=", "T",
    ]
    pcs = [pieces.Piece("p", 1, 3, 2, 0), pieces.Piece("t", 0, 0, 3, 3)]
    return (board, pcs)

def lvl16(): 
    board = [
        ".", ".", "t", ".", 
        ".", "=", "=", ".",
        "p", "=", ".", "P",
        "=", ".", "T", "=",
    ]
    pcs = [pieces.Piece("p", 2, 0, 2, 3), pieces.Piece("t", 0, 2, 3, 2)]
    return (board, pcs)

def lvl17(): 
    board = [
        ".", ".", "t", "T", 
        "=", "=", "=", ".",
        "P", ".", ".", ".",
        "=", ".", "=", "p",
    ]
    pcs = [pieces.Piece("p", 3, 3, 2, 0), pieces.Piece("t", 0, 2, 0, 3)]
    return (board, pcs)

def lvl18(): 
    board = [
        ".", ".", ".", ".", 
        ".", ".", "t", "T",
        ".", ".", ".", "=",
        "=", ".", "P", "p",
    ]
    pcs = [pieces.Piece("p", 3, 3, 3, 2), pieces.Piece("t", 1, 2, 1, 3)]
    return (board, pcs)

def lvl19(): # Level 19 -> 8 moves for perfect score
    board = [
        ".", "T", "=", ".", 
        "p", ".", "P", ".",
        ".", ".", ".", ".",
        "=", ".", "t", ".",
    ]
    pcs = [pieces.Piece("p", 1, 0, 1, 2), pieces.Piece("t", 3, 2, 0, 1)]
    return (board, pcs)

def lvl20(): # Level 20 -> 8 moves for perfect score
    board = [
        ".", "=", "=", ".", 
        "t", ".", ".", ".",
        "P", "=", "=", "p",
        "T", ".", ".", ".",
    ]
    pcs = [pieces.Piece("p", 2, 3, 2, 0), pieces.Piece("t", 1, 0, 3, 0)]
    return (board, pcs)

def lvl21(): # Level 21 -> 9 moves for perfect score
    board = [
        "T", ".", "p", "t", 
        "=", ".", "=", "P",
        "=", "=", ".", ".",
        ".", ".", "=", "=",
    ]
    pcs = [pieces.Piece("p", 0, 2, 1, 3), pieces.Piece("t", 0, 3, 0, 0)]
    return (board, pcs)

def lvl22(): # Level 22 -> 9 moves for perfect score
    board = [
        ".", "p", ".", "=", 
        "=", "t", ".", "T",
        "=", ".", "=", "P",
        "=", ".", "=", "=",
    ]
    pcs = [pieces.Piece("p", 0, 1, 2, 3), pieces.Piece("t", 1, 1, 1, 3)]
    return (board, pcs)
