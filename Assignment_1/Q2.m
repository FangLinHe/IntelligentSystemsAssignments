clear all;
close all;
clc

%% Parameters

% Set the maximum iteration timers.
max_size = 30000;

% training data and initial weights settings
input_pattern = ...
    [4, 2;
     4, 4;
     5, 3;
     5, 1;
     7, 2;
     1, 2;
     2, 1;
     3, 1;
     6, 5;
     3, 6;
     6, 7;
     4, 6;
     7, 6];

target = ...
   [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1];

test_input_pattern = ...
    [4, 1;
     5, 2;
     3, 4;
     5, 4;
     6, 1;
     7, 1;
     3, 2;
     8, 7;
     4, 7;
     7, 5;
     2, 3;
     2, 5];

test_target = ...
   [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1];

%initial_weights_hidden = zeros(3, 3); % for testing
initial_weights_hidden =  rand([3, 3]) / 5 - 0.1;
    % [w01 w02 w03;  --> for bias
    %  w11 w12 w13;  --> for x1
    %  w21 w22 w23;] --> for x2
    %  (h1)(h2)(h3)

%initial_weights_output = zeros(4, 1); % for testing
initial_weights_output = rand([4, 1]) / 5 - 0.1;
    % [w0;  --> for bias
    %  w1;  --> for h1
    %  w2;  --> for h2
    %  w3]  --> for h3
    %  (y)

%% Q2 (a) Plot the data. Would the network be able to solve this task if it had linear neurons only? Explain why.
Q2_a;

%% Q2(b) Implement and apply backpropagation (with£b= 1/30) until all examples are correctly classified. (This might take a few thousand epochs)
Q2_b;

%% Q2(c) Perform 10 runs with different random initializations of the weights, and plot the training and test error for each epoch averaged across the 10 runs. 
Q2_c;

%% Q2(d)Try different learning rates £b= [1, 1/3, 1/10, 1/30, 1/100, 1/300, 1/1000] and plot the training error over 1000 epochs.
Q2_d;

%% Q2(e) Vary the number of hidden units (from 1 to 10) and run each network with 10 different random initializations.
% It may take several minutes to run, since it has to run 10 different
% number of hidden units, and each has to run with 10 different initial
% weights.
Q2_e;
