# **Constructive Heuristics** for Vehicle Routing Problem with Time Windows (VRPTW)

#### **Problem** 

+ The VRPTW is a variant of VRP that has time windows constraints.
  + **Given:** A depot, a set of customers with coordinates, demands, time windows, and service times, a fleet of vehicles of the same capacity
  + **Objective:** Minimize the total travelling distances of all routes
  + **Constraints:** The vehicles start from the depot and return to the depot, each customer be visited once and only once, all the demands should be satisfied, the vehicle capacity should not be exceeded, the time window constraints should be satisfied

#### Algorithm Design Task

+ **Constructive heuristics** start from the depot and iteratively select the next unvisited customer. **The task** is to design the **heuristic** for selecting the next customer in each iteration.
  + **Inputs:** Current node, depot, unvisited nodes, demands of unvisited nodes, rest capacity of current vehicle, distance matrix, current time, time windows
  + **Outputs:** Next node

```{image} ./vrptw_construct.png
:width: 80%
:align: center
```

#### Evaluation

+ **Dataset:** Each designed algorithm is evaluated on 16 VRPTW instances. The number of customers in each instance is 50 and the coordinates are randomly sampled from [0,1], the demands are sampled from {1,2,...,9} and the capacity is 40. The time windows are sampled using the method in

+ **Fitness:** The average total distance over the 16 instances is used as the fitness in search.

#### Template: 

```python

template_program = '''
def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, current_time: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray, time_windows: np.ndarray) -> int:
    """Design a novel algorithm to select the next node in each step.
    Args:
        current_node: ID of the current node.
        depot: ID of the depot.
        unvisited_nodes: Array of IDs of unvisited nodes.
        rest_capacity: Rest capacity of vehicle
        current_time: Current time
        demands: Demands of nodes
        distance_matrix: Distance matrix of nodes.
        time_windows: Time windows of nodes.
    Return:
        ID of the next node to visit.
    """
    next_node = unvisited_nodes[0]
    return next_node
'''

task_description = "Help me design a novel algorithm to select the next node in each step."


```

