%% Q2(c) Perform 10 runs with different random initializations of the weights, and plot the training and test error for each epoch averaged across the 10 runs. 
if ~exist('max_size', 'var')
    max_size = 10000;
end
all_errors = zeros(max_size, 10);
all_correctness = zeros(max_size, 10);
classify_time = zeros(10, 1);
test_all_errors = zeros(max_size, 10);
test_all_correctness = zeros(max_size, 10);
test_classify_time = zeros(10, 1);
for i = 1:10
    [errors, correctness, correct_classify_time, test_errors, test_correctness, test_correct_classify_time] = ...
        multiple_neural(input_pattern, target, test_input_pattern, test_target, 3, 1/30, max_size);
    all_errors(1:length(errors), i) = errors;
    all_correctness(1:length(correctness), i) = correctness;
    classify_time(i) = correct_classify_time;
    test_all_errors(1:length(test_errors), i) = test_errors;
    test_all_correctness(1:length(test_correctness), i) = test_correctness;
    test_classify_time(i) = test_correct_classify_time;
    close(2);
end
all_errors_avg = mean(all_errors, 2);
all_correctness_avg = mean(all_correctness/13, 2);
classify_time_avg = mean(classify_time);
test_all_errors_avg = mean(test_all_errors, 2);
test_all_correctness_avg = mean(test_all_correctness/12, 2);
test_classify_time_avg = mean(test_classify_time);
figure;
subplot(2, 1, 1);
hold on;
plot(all_errors_avg, 'Color', 'b');
plot(test_all_errors_avg, 'Color', 'g');
legend('Error of training set', 'Error of test set');
xlabel('Iteration times');
ylabel('Mean square error');
hold off;
subplot(2, 1, 2);   
hold on;
plot(all_correctness_avg, 'Color', 'b');
plot(test_all_correctness_avg, 'Color', 'g');
legend('Correct rate of training set', 'Correct rate of test set');
xlabel('Iteration times');
ylabel('Correctly classified rate');
hold off;

fprintf('The average time to correctly classify all training points: %.4f\n', classify_time_avg);
fprintf('The average time to correctly classify all test points: %.4f\n', test_classify_time_avg);
