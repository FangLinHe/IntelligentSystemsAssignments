%% Q2(e) Vary the number of hidden units (from 1 to 10) and run each network with 10 different random initializations.
if ~exist('max_size', 'var')
    max_size = 10000;
end
all_errors = zeros(max_size, 10);
all_correctness = zeros(max_size, 10, 10);
test_all_errors = zeros(max_size, 10, 10);
test_all_correctness = zeros(max_size, 10, 10);
draw_order = [1 5 9 2 6 10 3 7 11 4];
figure;
for i = 1:10
    for j = 1:10
        [errors, correctness, correct_classify_time, test_errors, test_correctness, test_correct_classify_time] = ...
            multiple_neural(input_pattern, target, test_input_pattern, test_target, i, 1/30, max_size);
        close;

        all_errors(1:length(errors), i, j) = errors;
        all_correctness(1:length(correctness), i, j) = correctness;
        test_all_errors(1:length(test_errors), i, j) = test_errors;
        test_all_correctness(1:length(test_correctness), i, j) = test_correctness;
    end
    all_errors_avg = mean(sum(all_errors, 3), 2)/i;
    all_correctness_avg = mean(sum(all_correctness/13, 3), 2);
    test_all_errors_avg = mean(sum(test_all_errors, 3), 2)/i;
    test_all_correctness_avg = mean(sum(test_all_correctness/12, 3), 2);
    
    subplot(3, 4, draw_order(i));
    hold all;
    plot(all_errors_avg(1:end-1),  'Color', 'b');
    plot(test_all_errors_avg(1:end-1),  'Color', 'g');
    xlabel(sprintf('hidden layer size: %d', i));
    ylabel('Error');
    axis([0 max_size 0 8]);
end