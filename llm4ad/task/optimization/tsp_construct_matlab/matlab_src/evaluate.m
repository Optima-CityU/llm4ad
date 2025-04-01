function [avg_fitness, best_route, best_distance, best_matrix] = evaluate(num_cities, num_seeds)
%EVALUATE Generate multiple TSP instances and evaluate average fitness
%
%   Input:
%   num_cities      - (1,1) double, Number of cities in the problem
%   num_seeds       - (1,1) double, Number of different random seeds to test (default: 50)
%
%   Output:
%   avg_fitness     - (1,1) double, Average fitness across all seeds
%   best_route      - (1,:) double, Route array of the best solution found
%   best_distance   - (1,1) double, Total distance of the best solution
%   best_matrix     - (:,:) double, Distance matrix of the best solution
%
%   Example:
%   [avg_fit, best_route, best_dist, best_matrix] = evaluate(10)
%   [avg_fit, best_route, best_dist, best_matrix] = evaluate(10, 30)

    arguments (Input)
        num_cities (1,1) double {mustBeInteger, mustBePositive}
        num_seeds (1,1) double {mustBeInteger, mustBePositive} = 50
    end
    
    arguments (Output)
        avg_fitness (1,1) double {mustBeNonnegative}
        best_route (1,:) double {mustBeInteger, mustBePositive}
        best_distance (1,1) double {mustBeNonnegative}
        best_matrix (:,:) double {mustBeNumeric}
    end
    
    % Initialize arrays to store results
    all_distances = zeros(1, num_seeds);
    best_distance = inf;
    best_route = [];
    best_matrix = [];
    
    % Test multiple seeds
    for seed = 1:num_seeds
        % Set random seed for reproducibility
        rng(2025);
        
        % Generate random city coordinates
        coords = rand(num_cities, 2);
        
        % Calculate distance matrix
        distance_matrix = zeros(num_cities);
        for i = 1:num_cities
            for j = 1:num_cities
                if i ~= j
                    distance_matrix(i,j) = sqrt(sum((coords(i,:) - coords(j,:)).^2));
                end
            end
        end
        
        % Initialize route construction
        route = zeros(1, num_cities);
        route(1) = randi(num_cities);
        unvisited = setdiff(1:num_cities, route(1));
        
        % Construct route
        for i = 2:num_cities
            current_city = route(i-1); % current_node, destination_node, unvisited_nodes, distance_matrix
            next_city = tsp_construct(current_city, route(1), unvisited, distance_matrix);
            route(i) = next_city;
            unvisited = unvisited(unvisited ~= next_city);
        end
        
        % Calculate total distance
        total_distance = 0;
        for i = 1:num_cities-1
            total_distance = total_distance + distance_matrix(route(i), route(i+1));
        end
        % Add distance back to starting city
        total_distance = total_distance + distance_matrix(route(end), route(1));
        
        % Store the distance for this seed
        all_distances(seed) = total_distance;
        
        % Update best solution if current is better
        if total_distance < best_distance
            best_distance = total_distance;
            best_route = route;
            best_matrix = distance_matrix;
        end
    end
    
    % Calculate average fitness
    avg_fitness = mean(all_distances);
    
end
