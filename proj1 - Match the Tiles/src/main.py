from state import State
from pieces import Piece
import utils
import game_logic
import search_algorithms
import heuristics
import globals
import levels

from copy import deepcopy
import time
from memory_profiler import memory_usage

initial_state = list()

def main():

    while(True):
        print("Choose a Level between 1 and 25:")
        lvl = input("Level: ")
        if(int(lvl) > 0 and int(lvl) < 26): break

    lvl = getattr(levels, 'lvl' + str(lvl))
    (board, pieces) = lvl()


    while(True):
        print("[0] Player")
        print("[1] AI")
        play_choice = input("Game mode: ")
        if(int(play_choice) == 0 or int(play_choice) == 1): break

    if (play_choice == "0"):
        game_logic.player_loop(board, pieces)
        return

    debug_str = input("Show expanding nodes? (y/N)")
    if(debug_str == "y"): globals.debug = True
    
    utils.print_board(board)

    print("Using BFS:")
    start_mem = memory_usage()[0]
    start = time.time()
    bfs_sol = search_algorithms.bfs(board, pieces)
    end = time.time()
    end_mem = memory_usage()[0]
    bfs_exec_time =  (end - start)*1000
    bfs_nodes = globals.nodes_expanded
    bfs_mem_usage = (end_mem - start_mem)*1024

    print("Using DFS:")
    globals.nodes_expanded = 0 
    start_mem = memory_usage()[0]
    start = time.time()
    dfs_sol = search_algorithms.dfs(board, pieces)
    end = time.time()
    end_mem = memory_usage()[0]
    dfs_exec_time =  (end - start)*1000
    dfs_nodes = globals.nodes_expanded
    dfs_mem_usage = (end_mem - start_mem)*1024

    print("Using Iterative Deepening:")
    globals.nodes_expanded = 0 
    start_mem = memory_usage()[0]
    start = time.time()
    ids_sol = search_algorithms.iterative_deepening(board, pieces)
    end = time.time()
    end_mem = memory_usage()[0]
    ids_exec_time =  (end - start)*1000
    ids_nodes = globals.nodes_expanded
    ids_mem_usage = (end_mem - start_mem)*1024

    print("Using Greedy:")
    globals.nodes_expanded = 0
    start_mem = memory_usage()[0]
    start = time.time()
    greedy_sol = search_algorithms.a_star(board, pieces, heuristics.euclidean_distance)
    end = time.time()
    end_mem = memory_usage()[0]
    greedy_exec_time =  (end - start)*1000
    greedy_nodes = globals.nodes_expanded
    greedy_mem_usage = (end_mem - start_mem)*1024

    print("Using A*:")
    globals.nodes_expanded = 0
    start_mem = memory_usage()[0]
    start = time.time()
    a_star_sol = search_algorithms.a_star(board, pieces, heuristics.min_string)
    end = time.time()
    end_mem = memory_usage()[0]
    a_star_exec_time =  (end - start)*1000
    a_star_nodes = globals.nodes_expanded
    a_star_mem_usage = (end_mem - start_mem)*1024

    utils.print_table([("Alg.",          "Moves",        "Sol.",        "Exec Time(ms)",            "Nodes Exp.",        "Mem. Usage(KiB)"),
                 ("BFS",       str(len(bfs_sol)),   bfs_sol,    str(round(bfs_exec_time)),    str(bfs_nodes),      str(bfs_mem_usage)),                 
                 ("DFS",       str(len(dfs_sol)),   dfs_sol,    str(round(dfs_exec_time)),    str(dfs_nodes),      str(dfs_mem_usage)),
                 ("IDS",       str(len(ids_sol)),   ids_sol,    str(round(ids_exec_time)),    str(ids_nodes),      str(ids_mem_usage)),
                 ("Greedy",  str(len(greedy_sol)),  greedy_sol, str(round(greedy_exec_time)), str(greedy_nodes),   str(greedy_mem_usage)),
                 ("A*",      str(len(a_star_sol)),  a_star_sol, str(round(a_star_exec_time)), str(a_star_nodes),   str(a_star_mem_usage)),
    ])

main()