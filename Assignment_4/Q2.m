close all;
clear all;

% Load maze, rewards, status, actions
maze = importdata('Q2_map.txt', ' '); % 0: wall, 1: road, 2: start, 3: goal
R = importdata('Q2_reward.txt', ' ');
if size(maze) ~= size(R)
    error('Size of maze and rewards should be the same.');
end
S = get_S(maze);
A = [-1, 0; 0, 1; 1, 0; 0, -1]; % [N; E; S; W]

% parameters
para.discount_factor = 0.9; % discount factor
para.learning_rate = 0.4;
para.max_episode = 100;
para.max_actions = 100;
para.epsilon = 0.3;
para.prob_follow_act = 0.7;
para.path_filename = 'Q2_path.txt';

% Q learning and plot results
[Q, acum_reward] = Q_learning(maze, A, R, para);
plot(acum_reward);
