import heapq
import random

class ai:
	def __init__(self, num_pecas):
		self.num_pecas = num_pecas
		self.move_queue = []

		heapq.heappush(self.move_queue, (1, "w"))
		heapq.heappush(self.move_queue, (1, "s"))
		heapq.heappush(self.move_queue, (1, "a"))
		heapq.heappush(self.move_queue, (1, "d"))

		
	def a_star(self, movable, destination):
		cost = 1
		for i in range(0, len(movable)):
			conta = ((movable[i].row - destination[i].row)**2 + (movable[i].col - destination[i].col)**2)**1/2
			# print("movable row: {} col: {} dest row: {} col: {}".format(movable[i].row, movable[i].col, destination[i].row, destination[i].col))

			cost += conta
			# print(cost)
		
		return cost

	def no_heuristic(self):
		return 1

	
	# best_move -> [0] -> value ; [1] -> string
	def choose_move_horizontal(self, best_move, movable, destination):
		heapq.heappush(self.move_queue, (best_move[0] + self.a_star(movable, destination), best_move[1] + "a"))
		heapq.heappush(self.move_queue, (best_move[0] + self.a_star(movable, destination), best_move[1] + "d"))

	def choose_move_vertical(self, best_move, movable, destination):
		heapq.heappush(self.move_queue, (best_move[0] + self.a_star(movable, destination), best_move[1] + "w"))
		heapq.heappush(self.move_queue, (best_move[0] + self.a_star(movable, destination), best_move[1] + "s"))

	def choose_move(self, board, movable, destination):
		
		best_move = heapq.heappop(self.move_queue)
		
		if(best_move[1][-1] in ["w", "s"]):
			self.choose_move_horizontal(best_move, movable, destination)
			
		elif(best_move[1][-1] in ["a", "d"]):
			self.choose_move_vertical(best_move, movable, destination)

		return


	def get_best_move(self):
		return self.move_queue[0][1]
	
	def get_best_number_of_moves(self):
		return self.move_queue[0][0]

	def get_move_queue(self):
		return self.move_queue

	
	# def test_bot_move(self, mutable_pieces, destinationTiles):
	# 	print("In test_bot_move")









# if(best_move[1][-1] in ["w", "s"]):
# 			return
# 			#self.choose_move_horizontal(best_move, board, movable, destination)
# 		elif(best_move[1][-1] in ["a", "d"]):
# 			return
# 			#self.choose_move_vertical(best_move, board, movable, destination)