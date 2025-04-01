% 生成10个城市的TSP问题，测试50个不同的随机种子
[~, ~, ~, ~] = evaluate(10);

% 使用自定义的种子数量
[avg_fit, best_route, best_dist, best_matrix] = evaluate(50, 50);

% 显示最佳路径
% fprintf('Best route found: \n');
% fprintf('%d -> ', best_route(1:end-1));
% fprintf('%d\n', best_route(end));
fprintf('Avg: \n');
fprintf('%d\n', avg_fit)