from state import State
import game_logic

from heapq import heappush, heappop, heapify
from collections import deque
from copy import deepcopy


# Receives initial state.State and a list of pieces.Piece
# Does a Breadth First Search on the tree generated from the initial state
# Returns solution string with its moves
def bfs(start_state, pieces):
    
    explored, queue = set(), deque([State(start_state, None, "", 0, 0, 0, pieces)])

    while queue:

        node = queue.popleft()
        explored.add(node.map)

        if (game_logic.check_end(node.pieces)):
            print("Solution Found!")
            return node.move

        neighbours = game_logic.expand(node)

        for neighbour in neighbours:
            if neighbour.map not in explored:
                queue.append(neighbour)

# Receives initial state.State and a list of pieces.Piece
# Does a Depth First Search on the tree generated from the initial state
# Returns solution string with its moves
def dfs(start_state, pieces):

    explored, stack = set(), list([State(start_state, None, "", 0, 0, 0, pieces)])

    while stack:

        node = stack.pop()
        explored.add(node.map)

        if (game_logic.check_end(node.pieces)):
            print("Solution Found!")
            return node.move

        neighbors = reversed(game_logic.expand(node))

        for neighbor in neighbors:
            if neighbor.map not in explored:
                stack.append(neighbor)
                explored.add(neighbor.map)


# Receives initial state.State, a list of pieces.Piece, and a heuristic function
# Does a A* Search on the tree generated from the initial state, using the given heuristic function
# Returns solution string with its moves
def a_star(start_state, pieces, heuristic):

    explored, heap, heap_entry = set(), list(), {}

    root = State(start_state, None, "", 0, 0, 0, pieces)

    key = heuristic(root)

    root.key = key

    entry = (key, 0, root)

    heappush(heap, entry)

    heap_entry[root.map] = entry

    while heap:

        node = heappop(heap)
        explored.add(node[2].map)
        if (game_logic.check_end(node[2].pieces)):
            print("Solution Found!")
            return node[2].move

        neighbors = game_logic.expand(node[2])

        for neighbor in neighbors:

            
            neighbor.key = neighbor.cost + heuristic(neighbor)

            entry = (neighbor.key, neighbor.move, neighbor)

            if neighbor.map not in explored:

                heappush(heap, entry)

                explored.add(neighbor.map)

                heap_entry[neighbor.map] = entry

            elif neighbor.map in heap_entry and neighbor.key < heap_entry[neighbor.map][2].key:

                hindex = heap.index((heap_entry[neighbor.map][2].key,
                                     heap_entry[neighbor.map][2].move,
                                     heap_entry[neighbor.map][2]))

                heap[int(hindex)] = entry

                heap_entry[neighbor.map] = entry

                heapify(heap)


# Receives initial state.State and a list of pieces.Piece
# Does a Iterative Deepening Search on the tree generated from the initial state
# Returns solution string with its moves
def iterative_deepening(start_state, pieces):

    current_search_depth = 1

    original = deepcopy(start_state)

    while True:
        stack = list([State(original, None, "", 0, 0, 0, pieces)])
        while stack:
            node = stack.pop()
            if (game_logic.check_end(node.pieces)):
                print("Solution Found!")
                return node.move
            

            neighbors = reversed(game_logic.expand(node))

            for neighbor in neighbors:
                if node.depth < current_search_depth: 
                    stack.append(neighbor)
        current_search_depth += 1
