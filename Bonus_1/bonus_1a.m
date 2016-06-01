%load('mnist-original.mat'); % load the training set and test set.
random_pick_columns = randperm(70000); % generate random numbers 1~70,000 (shuffle)
training_data = double(data(:, random_pick_columns(1:6000)))'; % use the first 6,000 random number of data as training set
training_label_temp = label(1, random_pick_columns(1:6000))'; % training label
test_data = double(data(:, random_pick_columns(6001:7000)))'; % use the 6,001~7,000 random number of data as test data
test_label_temp = label(1, random_pick_columns(6001:7000))'; % test label