function next_node = tsp_construct(current_node, destination_node, unvisited_nodes, distance_matrix)
    % TSP_CONSTRUCT Select the next node in each step of TSP
    % 
    % Input:
    % current_node     - (1,1) double, ID of the current node
    % destination_node - (1,1) double, ID of the destination node
    % unvisited_nodes - (1,:) double, Array of IDs of unvisited nodes
    % distance_matrix - (:,:) double, Distance matrix of nodes
    % 
    % Output:
    % next_node       - (1,1) double, ID of the next node to visit
    % 
    % Example:
    % next = tsp_construct(1, 5, [2,3,4], rand(5,5))
    % Default implementation
    
    num_unvisited = length(unvisited_nodes);
    max_distance = -inf;
    next_node = unvisited_nodes(1);
    
    for j = 1:num_unvisited
        candidate_node = unvisited_nodes(j);
        distance_to_candidate = distance_matrix(current_node, candidate_node);
        
        if distance_to_candidate > max_distance
            max_distance = distance_to_candidate;
        end
    end
    
    for i = 1:num_unvisited
        candidate_node = unvisited_nodes(i);
        cost_to_candidate = distance_matrix(current_node, candidate_node);
        
        inverse_distance_score = 1 / (cost_to_candidate + 1e-6); % Adding a small value to avoid division by zero
        
        total_score = inverse_distance_score + max_distance;

        if total_score > next_node
            next_node = candidate_node;
        end
    end
end