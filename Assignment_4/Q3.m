close all;
clear all;

% Load maze, rewards, status, actions
maze = importdata('Q3_map.txt', ' '); % 0: wall, 1: road, 2: start, 3: goal
R = importdata('Q3_reward.txt', ' ');
if size(maze) ~= size(R)
    error('Size of maze and rewards should be the same.');
end
S = get_S(maze);
% actions: moving like a knight
A = [ 1,  2;
      1, -2;
     -1,  2;
     -1, -2;
      2,  1;
      2, -1;
     -2,  1;
     -2, -1];

% parameters
para.discount_factor = 0.9; % discount factor
para.learning_rate = 0.4;
para.max_episode = 100;
para.max_actions = 100;
para.epsilon = 0.3;
para.prob_follow_act = 0.72;
para.path_filename = 'Q3_path.txt';

% Q learning & plot
[Q, acum_reward] = Q_learning(maze, A, R, para);
plot(acum_reward)
