template_program = '''
function next_node = tsp_construct(current_node, destination_node, unvisited_nodes, distance_matrix)
%TSP_CONSTRUCT Select the next node in each step of TSP
%
%   Input:
%   current_node     - (1,1) double, ID of the current node
%   destination_node - (1,1) double, ID of the destination node
%   unvisited_nodes - (1,:) double, Array of IDs of unvisited nodes
%   distance_matrix - (:,:) double, Distance matrix of nodes
%
%   Output:
%   next_node       - (1,1) double, ID of the next node to visit
%
%   Example:
%   next = tsp_construct(1, 5, [2,3,4], rand(5,5))

    arguments (Input)
        current_node (1,1) double {mustBeInteger, mustBePositive}
        destination_node (1,1) double {mustBeInteger, mustBePositive}
        unvisited_nodes (1,:) double {mustBeInteger, mustBePositive}
        distance_matrix (:,:) double {mustBeNumeric}
    end
    
    arguments (Output)
        next_node (1,1) double {mustBeInteger, mustBePositive}
    end
    
    % Default implementation
    next_node = unvisited_nodes(1);
    
end
'''

task_description = "Given a set of nodes with their coordinates, you need to find the shortest route that visits each node once and returns to the starting node. \
The task can be solved step-by-step by starting from the current node and iteratively choosing the next node. Help me design a novel algorithm that is different from the algorithms in literature to select the next node in each step."
