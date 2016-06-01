close all;
clear all;

%% Q4 (A) Torus with four actions (NESW):
% Load the maze, rewards, status, actions
maze = importdata('Q4_map.txt', ' '); % 0: wall, 1: road, 2: start, 3: goal
R = importdata('Q4_reward.txt', ' ');
if size(maze) ~= size(R)
    error('Size of maze and rewards should be the same.');
end
S = get_S(maze); % status
A = [-1, 0; 0, 1; 1, 0; 0, -1]; % actions, representing [N; E; S; W]

% parameters
para.discount_factor = 0.9; % discount factor
para.learning_rate = 0.4;
para.max_episode = 100;
para.max_actions = 100;
para.epsilon = 0.1;
para.prob_follow_act = 0.72;
para.path_filename = 'Q4_path.txt';

% Q learning & plot results
[Q, acum_reward, exe_time_record] = Q_learning(maze, A, R, para);
plot(acum_reward)
figure;
plot(exe_time_record)

%% Q4 (B) Torus with eight actions (knight): 
% actions
A2 =[ 1,  2;
      1, -2;
     -1,  2;
     -1, -2;
      2,  1;
      2, -1;
     -2,  1;
     -2, -1];

% Q learning and plot results
[Q_A2, acum_reward_A2, exe_time_record_A2] = Q_learning(maze, A2, R, para);
figure;
plot(acum_reward_A2)
disp('Four directions: see Q');
disp('Eight directions: see Q_A2');
