# **Action Selection Strategy Heuristics** for **Cart Pole**

#### **Problem** 

The Cart Pole problem is a classic reinforcement learning task in OpenAI Gym, aiming to maximize the duration that a pole remains balanced on a moving cart within specific position and angle constraints.

```{image} ./cart.gif
:width: 80%
:align: center
```

+ **Given:** A cart with its position uniformly randomly assigned within the range (-0.05, 0.05), a pole stand on the cart.

+ **Objective:** Maximize the total number of iterations during which the pole angle remains within the range (-0.2095, 0.2095) (or ±12°).

+ **Constraints:** 
    - The position of the cart along the x-axis must be between (-2.4, 2.4).
    - The pole angle must between (-.2095, .2095).
    - The possible actions are:
        1. Push cart to the left
        2. Push cart to the right


#### Algorithm Design Task

+ **Action selection strategy heuristics:** Push the cart to move left and right iteratively, keeping the pole angle within the specified range. **The task** is to design the heuristic for selecting the action in each iteration.
  + **Inputs:** Cart position, cart velocity, pole angle, pole angular velocity, last selected action.
  + **Outputs:** Next action.

#### Evaluation

+ **Dataset:** Each designed algorithm is evaluated on gym environment. 

+ **Fitness:** 
    - If the pole can't stand within the iteration limit: `- final_iteration / max_iteration + 1 + 2.4`, where 1 is ratio bias and 2.4 is position bias.
    - If the pole stands within the iteration limit: `cart_position`.

#### Template: 

```python
template_program = '''
import numpy as np
def choose_action(cp: float, cv: float, pa: float, pav: float, last_action: int) -> int: 
    """
    Design a novel algorithm to select the action in each step.

    Args:
        cp: cart position, float between [-2.4, 2.4].
        cv: cart velocity, float between [-inf, inf].
        pa: pole angle, float between [-0.2095, 0.2095].
        pav: pole angular velocity, float between [-inf, inf].
        last_action: cart's next move, a int ranges between [0, 1, 2].

    Return:
         An integer representing the selected action for the cart.
         0: push cart to the left
         1: push cart to the right

    """
    # this is a placehold, replace it with your algorithm
    action =  np.random.randint(2)

    return action
'''

task_description = "I need help designing an innovative heuristic strategy function to prevent a pole from toppling over a cart, step by step. At each step, the function should select a specific action based on the pole's current state to move the cart, aiming to keep the pole balanced and upright without moving the cart too far from the center."



```

