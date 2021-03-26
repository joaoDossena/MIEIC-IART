import heapq
import random

class ai:
	def __init__(self, num_pecas):
		self.num_pecas = num_pecas
		self.move_queue = []
		heapq.heappush(self.move_queue, (0, 'w'))
		heapq.heappush(self.move_queue, (0, 'd'))
		heapq.heappush(self.move_queue, (0, 's'))
    	heapq.heappush(self.move_queue, (0, 'a'))
		
	def eval(self, board, movable, destination):
		
		return 0

	def choose_move_horizontal(self, past_moves):
		heapq.heappush(self.move_queue, (past_moves[0] + 1, past_moves[1] + "s"))
		heapq.heappush(self.move_queue, (past_moves[0] + 1, past_moves[1] + "w"))

	def choose_move_vertical(self, past_moves, board, movable, destination):
		heapq.heappush(self.move_queue, (past_moves[0] + 1, past_moves[1] + "a"))
		heapq.heappush(self.move_queue, (past_moves[0] + 1, past_moves[1] + "d"))

	def choose_move(self, past_moves, board, movable, destination):
		if(past_moves[1][-1] in ["w", "s"]):
			move = self.choose_move_horizontal(past_moves, board, movable, destination)
		elif(past_moves[1][-1] in ["a", "d"]):
			move = self.choose_move_vertical(past_moves, board, movable, destination)
		return move

