learning_rate = 1/300;
epochs_time = 50;

num_training_data = 6000;
num_test_data = 1000;

num_input_units = 784;
num_hidden_units = 100;
num_output_units = 10;

weights_input_hidden = rand([num_hidden_units, num_input_units+1]) / 5 - 0.1;

weights_hidden_output = rand([num_output_units, num_hidden_units+1]) / 5 - 0.1;

misclassifications = zeros(epochs_time, 1);
all_errors = zeros(epochs_time, 1);

test_misclassifications = zeros(epochs_time, 1);
test_all_errors = zeros(epochs_time, 1);

figure;

iter_time = 1;
while iter_time <= epochs_time
    fprintf('iteration %d\n', iter_time);
    sum_errors = 0;
    num_correct_classification = 0;
    
    test_sum_errors = 0;
    test_num_correct_classification = 0;
    for i = 1:num_training_data
        prediction_hidden = sigmoid(weights_input_hidden * [training_data(i, :)'; 1]); % weights_input_hidden(:, 785) for bias
        prediction_output = sigmoid(weights_hidden_output * [prediction_hidden; 1]);
        
        [~, predict_classification] = max(prediction_output);
        [~, target_classification] = max(training_label(i, :));
        num_correct_classification = num_correct_classification + (predict_classification == target_classification);
        
        target = training_label(i, :)';
        errors = (prediction_output - target).^2 / 2;
        sum_errors = sum_errors + sum(errors);
        
        delta_output = ((prediction_output - target) .* (1 - prediction_output) .* prediction_output);
        gradient_output_layer = delta_output * [prediction_hidden' 1];
        
        delta_hidden = (delta_output' * weights_hidden_output(:, 1:end-1))' .* (1 - prediction_hidden) .* prediction_hidden;
        gradient_hidden_layer = delta_hidden * [training_data(i, :) 1];
        
        weights_hidden_output = weights_hidden_output - learning_rate * gradient_output_layer;
        weights_input_hidden = weights_input_hidden - learning_rate * gradient_hidden_layer;
    end
    
    % TEST SET
    for i = 1:num_test_data
        test_prediction_hidden = sigmoid(weights_input_hidden * [test_data(i, :)'; 1]);
        test_prediction_output = sigmoid(weights_hidden_output * [test_prediction_hidden; 1]);
        
        [~, test_predict_classification] = max(test_prediction_output);
        [~, test_target_classification] = max(test_label(i, :));
        test_num_correct_classification = test_num_correct_classification + (test_predict_classification == test_target_classification);
        
        test_target = test_label(i, :)';
        test_errors = (test_prediction_output - test_target).^2 / 2;
        test_sum_errors = test_sum_errors + sum(test_errors);
    end
    
    all_errors(iter_time) = sum_errors / num_training_data;
    test_all_errors(iter_time) = test_sum_errors / num_test_data;
    
    misclassifications(iter_time) = (num_training_data - num_correct_classification) / num_training_data;
    test_misclassifications(iter_time) = (num_test_data - test_num_correct_classification) / num_test_data;
    fprintf('    error: %.4f\n', all_errors(iter_time));
    fprintf('    misclassification rate: %.4f\n', misclassifications(iter_time));
    fprintf('    [test] error: %.4f\n', test_all_errors(iter_time));
    fprintf('    [test] misclassification rate: %.4f\n', test_misclassifications(iter_time));
    
    subplot(2, 1, 1);
    hold all;
    plot(all_errors(1:iter_time), 'g');
    plot(test_all_errors(1:iter_time), 'b');
    axis([1 iter_time+1 0 all_errors(1)+0.05]);
    xlabel('Epochs');
    ylabel('Error');
    legend('Training set', 'Test set');
    hold off;
    
    subplot(2, 1, 2);
    hold all;
    plot(misclassifications(1:iter_time), 'g');
    plot(test_misclassifications(1:iter_time), 'b');
    axis([1 iter_time+1 0 misclassifications(1)+0.05]);
    xlabel('Epochs');
    ylabel('Misclassification rate');
    hold off;
    
    drawnow;
    
    iter_time = iter_time + 1;
end