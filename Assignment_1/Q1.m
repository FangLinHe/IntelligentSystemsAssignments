clear all;
close all;
clc
%% Q1 (a) Plot the training set with the initial separation line where the two classes are displayed in different colours.
% training data and initial weights settings
input_pattern = ...
    [1, 8;
     6, 2;
     3, 6;
     4, 4;
     3, 1;
     1, 6;
     6, 10;
     7, 7;
     6, 11;
     10, 5;
     4, 11];

target = ...
   [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1];

initial_weights = ...
    [0.8 -0.5 2]; % 0.8 for x1, -0.5 for x2, 2 for bias

% plot training data
hold on;
plot(input_pattern(1:6, 1), input_pattern(1:6, 2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
plot(input_pattern(7:11, 1), input_pattern(7:11, 2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'g');

% plot initial separation line
adjustvalue = 10;
x11 = min(input_pattern(:, 1)) - adjustvalue;
x12 = -1 * (initial_weights(1) * x11 + initial_weights(3)) / initial_weights(2);
x21 = max(input_pattern(:, 1)) + adjustvalue;
x22 = -1 * (initial_weights(1) * x21 + initial_weights(3)) / initial_weights(2);
line([x11 x21], [x12 x22], 'LineStyle', '--', 'LineWidth', 3, 'Color', 'r');

% set axis parameters
axis([0 12 0 12]);
legend('target = 1', 'target = -1');
title('Q1 (a) Before training');
xlabel('x1');
ylabel('x2');


%% Q1 (b)(c) Implement the delta rule and apply it for all points from the test set, with a learning rate of £b= 1/50. Plot the new separation line.

figure;

% plot training data
hold on;
plot(input_pattern(1:6, 1), input_pattern(1:6, 2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
plot(input_pattern(7:11, 1), input_pattern(7:11, 2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'g');

% plot initial separation line
adjustvalue = 10;
x11 = min(input_pattern(:, 1)) - adjustvalue;
x12 = -1 * (initial_weights(1) * x11 + initial_weights(3)) / initial_weights(2);
x21 = max(input_pattern(:, 1)) + adjustvalue;
x22 = -1 * (initial_weights(1) * x21 + initial_weights(3)) / initial_weights(2);
line([x11 x21], [x12 x22], 'LineStyle', '--', 'LineWidth', 3, 'Color', 'r');

% set axis parameters
axis([0 12 0 12]);
legend('target = 1', 'target = -1');
title('Q1 (b)(c) After applying delta rule');
xlabel('x1');
ylabel('x2');

single_neural(input_pattern, target, initial_weights, 0.02, 1);
close(3);


%% Q1 (d) Train the perceptron until all the points are correctly classified, and plot the final decision boundary line.

figure;

% plot training data
hold on;
plot(input_pattern(1:6, 1), input_pattern(1:6, 2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
plot(input_pattern(7:11, 1), input_pattern(7:11, 2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'g');

% plot initial separation line
adjustvalue = 10;
x11 = min(input_pattern(:, 1)) - adjustvalue;
x12 = -1 * (initial_weights(1) * x11 + initial_weights(3)) / initial_weights(2);
x21 = max(input_pattern(:, 1)) + adjustvalue;
x22 = -1 * (initial_weights(1) * x21 + initial_weights(3)) / initial_weights(2);
line([x11 x21], [x12 x22], 'LineStyle', '--', 'LineWidth', 3, 'Color', 'r');

% set axis parameters
axis([0 12 0 12]);
legend('target = 1', 'target = -1');
title('Q1 (d)Train until all the points are correctly classified');
xlabel('x1');
ylabel('x2');

new_weights = single_neural(input_pattern, target, initial_weights, 0.02, -1);

hold off;

figure;
hold on;
plot(input_pattern(1:6, 1), input_pattern(1:6, 2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
plot(input_pattern(7:11, 1), input_pattern(7:11, 2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'g');

% plot initial separation line
adjustvalue = 10;
x11 = min(input_pattern(:, 1)) - adjustvalue;
x12 = -1 * (new_weights(1) * x11 + new_weights(3)) / new_weights(2);
x21 = max(input_pattern(:, 1)) + adjustvalue;
x22 = -1 * (new_weights(1) * x21 + new_weights(3)) / new_weights(2);
line([x11 x21], [x12 x22], 'LineStyle', '--', 'LineWidth', 3, 'Color', 'r');

% set axis parameters
axis([0 12 0 12]);
legend('target = 1', 'target = -1');
title('Q1 (d) Plot the final decision boundary line');
xlabel('x1');
ylabel('x2');

%% Q1 (e) Is it a good solution? Discuss potential problems that may arise.

figure;

% plot training data
hold on;
plot(input_pattern(1:6, 1), input_pattern(1:6, 2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
plot(input_pattern(7:11, 1), input_pattern(7:11, 2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'g');

% plot initial separation line
adjustvalue = 10;
x11 = min(input_pattern(:, 1)) - adjustvalue;
x12 = -1 * (initial_weights(1) * x11 + initial_weights(3)) / initial_weights(2);
x21 = max(input_pattern(:, 1)) + adjustvalue;
x22 = -1 * (initial_weights(1) * x21 + initial_weights(3)) / initial_weights(2);
line([x11 x21], [x12 x22], 'LineStyle', '--', 'LineWidth', 3, 'Color', 'r');

% set axis parameters
axis([0 12 0 12]);
legend('target = 1', 'target = -1');
title('Q1 (e) My solution to better training');
xlabel('x1');
ylabel('x2');

new_weights = single_neural(input_pattern, target, initial_weights, 0.02, 100);


hold off;

figure;
hold on;
plot(input_pattern(1:6, 1), input_pattern(1:6, 2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
plot(input_pattern(7:11, 1), input_pattern(7:11, 2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'g');

% plot initial separation line
adjustvalue = 10;
x11 = min(input_pattern(:, 1)) - adjustvalue;
x12 = -1 * (new_weights(1) * x11 + new_weights(3)) / new_weights(2);
x21 = max(input_pattern(:, 1)) + adjustvalue;
x22 = -1 * (new_weights(1) * x21 + new_weights(3)) / new_weights(2);
line([x11 x21], [x12 x22], 'LineStyle', '--', 'LineWidth', 3, 'Color', 'r');

% set axis parameters
axis([0 12 0 12]);
legend('target = 1', 'target = -1');
title('Q1 (e) My better separation line');
xlabel('x1');
ylabel('x2');
