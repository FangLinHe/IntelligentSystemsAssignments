%% Q2(d)Try different learning rates £b= [1, 1/3, 1/10, 1/30, 1/100, 1/300, 1/1000] and plot the training error over 1000 epochs.
max_size = 1000;
all_errors = zeros(max_size, 7);
all_correctness = zeros(max_size, 7);
classify_time = zeros(7, 1);
eta = [1, 1/3, 1/10, 1/30, 1/100, 1/300, 1/1000];
eta_string = {'1', '1/3', '1/10', '1/30','1/100', '1/300', '1/1000'};
figure;
xlabel('Iteration times');
ylabel('Mean square error');
for i = 1:7
    [errors, correctness, correct_classify_time] = ...
        multiple_neural(input_pattern, target, test_input_pattern, test_target, 3, eta(i), max_size);
    all_errors(1:length(errors), i) = errors;
    all_correctness(1:length(correctness), i) = correctness;
    classify_time(i) = correct_classify_time;
    close;
    hold all;
    plot(errors(1:end-1), 'DisplayName', eta_string{i});
    legend('-DynamicLegend');
end
