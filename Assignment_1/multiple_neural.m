function [training_errors, training_correctness, correct_classify_time, test_errors, test_correctness, test_correct_classify_time] = ...
    multiple_neural(input_pattern, target, test_input_pattern, test_target, size_of_hidden_layer, learning_rate, max_iteration_time)
% Assume the dimension of the neural network is N, and there are totally P
% training set.
% Input:
%    input_pattern: training set, PxN matrix, with the form of:
%       [x11, x12, ... , x1N;
%        x21, x22, ... , x2N;
%        ...;
%        xP1, xP2, ... , xPN ]
%    target: training set, P dimensional vector with the form of:
%       [t1, t2, ... , tP]
%    test_input_pattern: test set, PxN matrix, with the form of:
%       [x11, x12, ... , x1N;
%        x21, x22, ... , x2N;
%        ...;
%        xP1, xP2, ... , xPN ]
%    test_target: test set, P dimensional vector with the form of:
%       [t1, t2, ... , tP]
%    initial_weights_output: H+1 dimensional vector with the form of:
%       [w0;  --> for bias
%        w1;  --> for h1
%        w2;  --> for h2
%        w3]  --> for h3
%        (y)
%    learning_rate: scalar value; default value is 0.2.
%    max_iteration_time: scalar value; if this value is less than 1, it
%       will train until all the training set is classified correctly.
%       Default value is 10000.
%    error_stop_threshold: stop threshold, a scalar, exit the loop when the
%       difference between the errors in this iteration from the previous
%       iteration is less than error_stop_threshold.

P = size(input_pattern, 1); % the number of training data, in this assignment, 13
N = size(input_pattern, 2); % the size of input data, in this assignment, 2

if nargin < 3
    test_input_pattern = input_pattern;
end
if nargin < 4
    test_target = target;
end
test_P = size(test_input_pattern, 1);
if nargin < 5
    size_of_hidden_layer = 3;
end
if nargin < 6
    learning_rate = 1/30;
end
if nargin < 7
    max_iteration_time = -1;
end
stop_when_classify_correctly = (max_iteration_time <= 0);


initial_weights_hidden =  rand([N+1, size_of_hidden_layer]) / 5 - 0.1;
%    H = size_of_hidden_layer
%    initial_weights_hidden: P+1xH dimensional matrix with the form of:
%       [w01 w02 ... w0H;  --> for bias
%        w11 w12 ... w1H;  --> for x1
%        ...;
%        wN1 wN2 ... wNH;] --> for xN
%        (h1)(h2)... (hH)
%       w01, w11, ... are the weight of bias for h1, h2, ...

initial_weights_output = rand([size_of_hidden_layer+1, 1]) / 5 - 0.1;
    % [w0;  --> for bias
    %  w1;  --> for h1
    %  w2;  --> for h2
    %  w3]  --> for h3
    %  (y)

training_errors = zeros(max_iteration_time, 1);
training_correctness = zeros(max_iteration_time, 1);
test_errors = zeros(max_iteration_time, 1);
test_correctness = zeros(max_iteration_time, 1);
iter_time = 1;

%{
fprintf('Initial weights for input & hidden layer:\n');
fprintf('[w01 w02 w03;  --> for bias\n');
fprintf(' w11 w12 w13;  --> for x1\n');
fprintf(' w21 w22 w23;] --> for x2\n');
fprintf(' (h1)(h2)(h3)\n');
disp(initial_weights_hidden);
fprintf('Initial weights for hidden layer & output layer:\n');
fprintf('%f  --> for bias\n', initial_weights_output(1));
fprintf('%f  --> for h1\n', initial_weights_output(2));
fprintf('%f  --> for h2\n', initial_weights_output(3));
fprintf('%f  --> for h3\n', initial_weights_output(4));
fprintf('\n---------------------------------------------------------------\n');
%}

correct_classify_time = 0;
test_correct_classify_time = 0;
while iter_time < max_iteration_time || stop_when_classify_correctly
    %fprintf('Iteration %d:\n', iter_time);
    
    % Calculate activation of each hidden node and store them
    bias_and_input_pattern = [ones(P, 1) input_pattern]; % [bais=1 x11 x12; 1 x21 x22; ...]
    hidden_layer = [ones(P, 1) sigmoid(bias_and_input_pattern * initial_weights_hidden)]; %  [bias=1 h11 h12 h13; 1 h21 h22 h23; ...]
    
    % TEST SET: Calculate activation of each hidden node and store them
    test_bias_and_input_pattern = [ones(test_P, 1) test_input_pattern]; % [bais=1 x11 x12; 1 x21 x22; ...]
    test_hidden_layer = [ones(test_P, 1) sigmoid(test_bias_and_input_pattern * initial_weights_hidden)]; %  [bias=1 h11 h12 h13; 1 h21 h22 h23; ...]
    

    % Calculate activation of each output node
    y = hidden_layer * initial_weights_output;
    
    % TEST SET: Calculate activation of each output node
    test_y = test_hidden_layer * initial_weights_output;
    
    % Calculate square error
    square_error = sum((target' - y).^2 / 2);
    %fprintf('    Squared error: %f\n', square_error);
    training_errors(iter_time) = square_error;
    
    % TEST SET: Calculate square error
    test_square_error = sum((test_target' - test_y).^2 / 2);
    %fprintf('    Squared error: %f\n', square_error);
    test_errors(iter_time) = test_square_error;

    %
    correctness_number = sum(((y > 0) == (target' > 0))*1);
    %fprintf('    Correct classified number: %f\n', correctness_number);
    training_correctness(iter_time) = correctness_number;
    %fprintf('    Current classified:\n');
    %disp((y' > 0) * 2 - 1);

    % TEST SET:
    test_correctness_number = sum(((test_y > 0) == (test_target' > 0))*1);
    %fprintf('    Correct classified number: %f\n', test_correctness_number);
    test_correctness(iter_time) = test_correctness_number;
    %fprintf('    Current classified:\n');
    %disp((y' > 0) * 2 - 1);

    % Calculation of gradient for output layer
    derivative_error_w_h_y = (y' - target) * hidden_layer;
    

    % Calculation of gradient for hidden layer
    derivative_error_w_x_h = (((y - target') * initial_weights_output(2:size_of_hidden_layer+1)') .* (hidden_layer(:, 2:size_of_hidden_layer+1) .* (1-hidden_layer(:, 2:size_of_hidden_layer+1))))' * bias_and_input_pattern;


        % [w01 w02 w03;  --> for bias
        %  w11 w12 w13;  --> for x1
        %  w21 w22 w23;] --> for x2
        %  (h1)(h2)(h3)

    initial_weights_hidden = initial_weights_hidden - derivative_error_w_x_h' * learning_rate / P;
        % [w0;  --> for bias
        %  w1;  --> for h1
        %  w2;  --> for h2
        %  w3]  --> for h3
        %  (y)
    initial_weights_output_dif = derivative_error_w_h_y' * learning_rate / P;
    initial_weights_output = initial_weights_output - initial_weights_output_dif;
    
    if test_correct_classify_time == 0 && test_correctness_number == test_P-1
        test_correct_classify_time = iter_time;
    end
    if correct_classify_time == 0 && correctness_number == P
        correct_classify_time = iter_time;
        if stop_when_classify_correctly
            break;
        end
    end
    iter_time = iter_time + 1;
    
end

figure;
subplot(2, 2, 1);
plot(training_errors);
subplot(2, 2, 2);
plot(training_correctness);
subplot(2, 2, 3);
plot(test_errors);
subplot(2, 2, 4);
plot(test_correctness);

%{
fprintf('Total iteration time: %d\n', iter_time);
fprintf('Final weights for input & hidden layer:\n');
fprintf('[w01 w02 w03;  --> for bias\n');
fprintf(' w11 w12 w13;  --> for x1\n');
fprintf(' w21 w22 w23;] --> for x2\n');
fprintf(' (h1)(h2)(h3)\n');
disp(initial_weights_hidden);
fprintf('Final weights for hidden layer & output layer:\n');
fprintf('%f  --> for bias\n', initial_weights_output(1));
fprintf('%f  --> for h1\n', initial_weights_output(2));
fprintf('%f  --> for h2\n', initial_weights_output(3));
fprintf('%f  --> for h3\n', initial_weights_output(4));

fprintf('Final y:\n');
disp(y');
fprintf('Final squared error: %f\n', square_error);
%}
end % function multiple_neural
