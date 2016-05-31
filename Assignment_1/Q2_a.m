%% Q2 (a) Plot the data. Would the network be able to solve this task if it had linear neurons only? Explain why.
% plot training data
hold on;
plot(input_pattern(1:5, 1), input_pattern(1:5, 2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
plot(input_pattern(6:13, 1), input_pattern(6:13, 2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
plot(test_input_pattern(1:6, 1), test_input_pattern(1:6, 2), 'x', 'MarkerSize', 8, 'Color', 'b');
plot(test_input_pattern(7:12, 1), test_input_pattern(7:12, 2), 'x', 'MarkerSize', 8, 'Color', 'g');

% set axis parameters
axis([0 10 0 10]);
legend('target = 1', 'target = -1', 'test target = 1', 'test target = -1');
title('Q1 (a) Before training');
xlabel('x1');
ylabel('x2');
