class ai:

	def __init__():
		return

	eval(board, movable, destination):
		return 0

	choose_move(last_move, board, movable, destination):
		last_move.lower()
		if(last_move in ["w", "s"]):
			choose_move_horizontal(last_move, board, movable, destination)
		else:
			choose_move_vertical(last_move, board, movable, destination)
		return

	choose_move_horizontal(last_move, board, movable, destination):
		return
	choose_move_vertical(last_move, board, movable, destination):
		return