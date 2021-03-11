# MIEIC-IART

João Dossena - UP201800174  
João Rosário - UP201806334  
João Sousa - UP201806613 

## Theme: Topic 1 - Heuristic Search Methods for One Player Solitaire Games
Our chosen Solitaire Game is Match the Tiles. 

## Checkpoint
### 1. Specification of the work to be performed (definition of the game)
Swipe (Up, Down, Left, Right) to move the tiles. You should place all colored tiles onto the designated spots with the same color. The moves are synchronized, thus you must use existing fixed tiles to create gaps between tiles and solve the puzzle.

MATCH TILES - SLIDING PUZZLE GAME FEATURES:
-  Different levels of varying complexity.
-  Normal (4x4), big (5x5) and bigger (6x6) board sizes.
-  Undo moves support.
-  No time limit.

### 2. Related work with references to works found in a bibliographic search (articles, web pages and/or source code)
-  [The original game](https://play.google.com/store/apps/details?id=net.bohush.match.tiles.color.puzzle&hl=pt_PT&gl=US)
-  []()

### 3. Formulation of the problem as a search problem
#### 3.1. State Representation:
The game state can be represented by a matrix with characters representing free slots, obstacles, and the colored final slots (EXEMPLIFICAR QUAIS CARACTERES REPRESENTAM CADA COISA)
#### 3.2. Initial State:
The initial state will be generated randomly so that the game is not always the same. However, for explanation purposes we will assume a specific initial state. (EXEMPLIFICAR QUAL O ESTADO PARA NOSSO RELATORIO)
#### 3.3. Objective Test:
The objective test has to verify if all of the movable pieces are in their respective desired spots. (EXEMPLIFICAR MELHOR)
#### 3.4. Operators (names, preconditions, effects and costs):
(MELHORAR PRECONDITIONS?)  
Operator |    Preconditions   |                     Effects                     | Cost
------
UP       | Last move != UP    | All movable pieces move up until an obstacle    | 1
DOWN     | Last move != DOWN  | All movable pieces move down until an obstacle  | 1
LEFT     | Last move != LEFT  | All movable pieces move left until an obstacle  | 1
RIGHT    | Last move != RIGHT | All movable pieces move right until an obstacle | 1
#### 3.5. Heuristics/Evaluation Function:

### 4. Implementation work already carried out 
#### 4.1. Programming language

#### 4.2. Development environment

#### 4.3. Data structures

#### 4.4. File structure

#### 4.5. Other stuff

---
## Final delivery
