function out = single_neural(input_pattern, target, initial_weights, learning_rate, max_iteration_time, error_stop_threshold)
% Assume the dimension of the neural network is N, and there are totally P
% training set.
% Input:
%    input_pattern: PxN matrix, with the form of:
%       [x11, x12, ... , x1N;
%        x21, x22, ... , x2N;
%        ...;
%        xP1, xP2, ... , xPN ]
%    target: P dimensional vector with the form of:
%       [t1, t2, ... , tP]
%    initial_weights: N+1 dimensional vector with the form of:
%       [w1, w2, ... , wN, wB]
%       wB is the weight of bias.
%    learning_rate: scalar value; default value is 0.2.
%    max_iteration_time: scalar value; if this value is less than 1, it
%       will train until all the training set is classified correctly.
%       Default value is 100.
%    error_stop_threshold: stop threshold, exit the loop when the
%       difference between the errors in this iteration from the previous
%       iteration is small enough.

P = size(input_pattern, 1);
N = size(input_pattern, 2);
if nargin < 3
    initial_weights = rand([1 N+1]) -0.5;
end
if nargin < 4
    learning_rate = 0.02;
end
if nargin < 5
    max_iteration_time = 100;
end
stop_when_classify_correctly = (max_iteration_time <= 0);
if nargin < 6
    error_stop_threshold = 0.001;
end
if length(target) ~= P
    error('Dimension of target is not identical to input pattern.');
end
if length(initial_weights) ~= N+1
    error('Dimension of target is not identical to input pattern.');
end


weights = repmat(initial_weights, [P 1]);
bias = weights(1, N+1);
net = sum(input_pattern .* weights(:, 1:N), 2) + bias;
square_error = sum((target' - net).^2 / 2);
fprintf('Initial settings:\n');
fprintf('    Squared error: %f\n', square_error);
fprintf('    Weights: \n');
disp(initial_weights);

iteration_times = 0;
error_record = zeros(max_iteration_time, 1);
while iteration_times < max_iteration_time || stop_when_classify_correctly
    % Calculate net
    weights = repmat(initial_weights, [P 1]);
    bias = weights(1, N+1);
    net = sum(input_pattern .* weights(:, 1:N), 2) + bias;

    % Obtain the partial derivative of error with respect to each weight
    % [dE/dw_1, dE/dw_2, ... , dE/dw_N]
    derivative_error = sum(repmat(net - target', [1 N]) .* input_pattern, 1) / P;

    % Change weights in the opposite direction of dE/dw_i
    delta_weight = -1 * (learning_rate) .* derivative_error;
    delta_bias = -1 * (learning_rate) .* sum(net - target') / P;
    initial_weights(1:N) = initial_weights(1:N) + delta_weight;
    initial_weights(N+1) = initial_weights(N+1) + delta_bias;
    
    iteration_times = iteration_times + 1;

    
    % Calculate new net
    weights = repmat(initial_weights, [P 1]);
    bias = weights(1, N+1);
    net = sum(input_pattern .* weights(:, 1:N), 2) + bias;
    
    % Calculate squared error between output and target
    new_square_error = sum((target' - net).^2 / 2);
    
    % Make sure the current settings do not cause divergence.
    if new_square_error > square_error
        error('Diverge! Please set smaller learning_rate.');
    % If the error difference is too small, exit the loop
    elseif square_error - new_square_error < error_stop_threshold
        break;
    end
    
    % Record the current error
    square_error = new_square_error;
    error_record(iteration_times) = square_error;
    
    % Calculate correctness rate
    correctness_rate = sum(((input_pattern * initial_weights(1:N)' + bias > 0) == (target' > 0))*1) / P;
    
    % Display the current settings and error
    fprintf('Iteration %d:\n', iteration_times);
    fprintf('    Correctness rate: %f\n', correctness_rate);
    fprintf('    Squared error: %f\n', square_error);
    fprintf('    Weight: \n');
    disp(initial_weights);
    
    % Plot the new separation line
    adjustvalue = 10;
    x11 = min(input_pattern(:, 1)) - adjustvalue;
    x12 = -1 * (initial_weights(1) * x11 + initial_weights(3)) / initial_weights(2);
    x21 = max(input_pattern(:, 1)) + adjustvalue;
    x22 = -1 * (initial_weights(1) * x21 + initial_weights(3)) / initial_weights(2);
    
    line([x11 x21], [x12 x22], 'Color', 'c');
    drawnow;
    
    % Stop when all training sets are correctly classified
    if stop_when_classify_correctly && correctness_rate == 1
        break;
    end
    
end
figure,
plot(error_record(1:iteration_times-1));
title('Error trends');
xlabel('Iteration times');
ylabel('Error');
out = initial_weights;
