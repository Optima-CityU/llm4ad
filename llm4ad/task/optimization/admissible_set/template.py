
template_program = '''
import numpy as np

def priority(el: tuple, n: int, w: int) -> np.array:
    """Design a novel algorithm to evaluate a vector for potential inclusion in a set
    Args:
        el: Candidate vectors for the admissible set.
        n: Number of dimensions and the length of a vector.
        w: Weight of each vector.

    Return:
        The priorities of `el`.
    """
    priorities = sum(abs(i) for i in el) / n
    return priorities
'''

task_description = """
Help me design a novel algorithm to evaluate vectors for potential inclusion in a set. 
This involves iteratively scoring the priority of adding a vector 'el' to the set based on analysis (like bitwise), 
with the objective of maximizing the set's size.
"""

