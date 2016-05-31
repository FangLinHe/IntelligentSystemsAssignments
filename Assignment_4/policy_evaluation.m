function [V, deltas] = policy_evaluation(maze, A, P, R, pi, para)
if nargin >= 7
    if ~isfield(para, 'stop_thres')
        para.stop_thres = 0.001;
    end
    if ~isfield(para, 'discount_factor')
        para.discount_factor = 0.9;
    end
end
V = zeros(size(maze));
epochs = 0;
max_epochs = 100;
deltas = zeros(max_epochs, 1);
while epochs < max_epochs
    delta = 0;
    v = V; % value backup
    for i = 1:size(maze, 1)
        for j = 1:size(maze, 2)
            if maze(i, j)
                %v = V(i, j); % value backup
                V(i, j) = get_new_value([i, j], v, A, P, R, pi, para);
                delta = max(delta, abs(v(i, j) - V(i, j)));
            end
        end
    end
    epochs = epochs + 1;
    deltas(epochs) = delta;
    fprintf('%d: %f\n', epochs, delta);
    if delta < para.stop_thres
        break;
    end
end
plot(deltas(1:epochs));



function sum_a = get_new_value(s1, V, A, P, R, pi, para)
sum_a = 0;
for a = 1:size(A, 1)
    sum_a = sum_a + pi(s1(1), s1(2), a) * Q_function(s1, a, V, P, R, para);
end

function Q = Q_function(s1, a, V, P, R, para)
Q = 0;
for i = max(1, s1(1)-1):min(size(V, 1), s1(1)+1)
    for j = max(1, s1(2)-1):min(size(V, 2), s1(2)+1)
        Q = Q + P(s1(1), s1(2), a, i, j) * (R(i, j) + para.discount_factor * V(i, j));
    end
end