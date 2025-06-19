# **Constructive Heuristics** for Capacitated Vehicle Routing Problem (CVRP)

#### **Problem** 

+ The **Capacitated Vehicle Routing Problem (CVRP)** is one of the most significant and well-studied combinatorial optimization problems. 
  + **Given:** A depot, a set of customers with coordinates and demands, a fleet of vehicles of the same capacity
  + **Objective:** Minimize the total travelling distances of all routes
  + **Constraints:** The vehicles start from the depot and return to depot, each customer be visited once and only once, all the demands should be satisfied, the capacity of vehicle should not be exceeded

#### Algorithm Design Task

+ **Constructive heuristics** start from the depot and iteratively select the next unvisited customer. **The task** is to design the **heuristic** for selecting the next customer in each iteration.
  + **Inputs:** Current node, unvisited nodes, demands of unvisited nodes, rest capacity of current vehicle, distance matrix
  + **Outputs:** Next node

```{image} ./cvrp_construct.png
:width: 80%
:align: center
```

#### Evaluation

+ **Dataset:** Each designed algorithm is evaluated on 16 CVRP instances. The number of customers in each instance is 50 and the coordinates are randomly sampled from [0,1], the demands are sampled from {1,2,...,9} and the capacity is 40.

+ **Fitness:** The average total distance over the 16 instances is used as the fitness in search.

#### Template: 

```python

template_program = '''
import numpy as np

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    """Design a novel algorithm to select the next node in each step.
    Args:
        current_node: ID of the current node.
        depot: ID of the depot.
        unvisited_nodes: Array of IDs of unvisited nodes.
        rest_capacity: rest capacity of vehicle
        demands: demands of nodes
        distance_matrix: Distance matrix of nodes.
    Return:
        ID of the next node to visit.
    """
    next_node = unvisited_nodes[0]
    return next_node
'''

task_description = "Help me design a novel algorithm to select the next node in each step."

```

