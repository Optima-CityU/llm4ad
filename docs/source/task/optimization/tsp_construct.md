# **Constructive Heuristics** for **Traveling Salesman Problem (TSP)**

#### **Problem** 

+ The **Traveling Salesman Problem (TSP)** is one of the most significant and well-studied combinatorial optimization problems. 
  + **Given:** A set of cities with coordinates or distance matrix, a salesman
  + **Objective:** Minimize the total travelling distance 
  + **Constraints:** Each city must be visited once and only once

#### Algorithm Design Task

+ **Constructive heuristics** start from one city and iteratively select the next unvisited city. **The task** is to design the **heuristic** for selecting the next city in each iteration.
  + **Inputs:** Current city, destination city, unvisited city, distance matrix
  + **Outputs:** Next city

```{image} ./tsp_construct.png
:width: 80%
:align: center
```


#### Evaluation

+ **Dataset:** Each designed algorithm is evaluated on 16 TSP instances. The number of cities in each instance is 50 and the coordinates are randomly sampled from [0,1]. 

+ **Fitness:** The average distance of the route over the 16 instances is used as the fitness in search.

#### Template: 

```python
template_program = '''
import numpy as np
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int: 
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    # this is a placehold, replace it with your algorithm
    next_node = unvisited_nodes[0]

    return next_node
'''

task_description = "Given a set of nodes with their coordinates, you need to find the shortest route that visits each node once and returns to the starting node. The task can be solved step-by-step by starting from the current node and iteratively choosing the next node. Help me design a novel algorithm that is different from the algorithms in literature to select the next node in each step."

```

