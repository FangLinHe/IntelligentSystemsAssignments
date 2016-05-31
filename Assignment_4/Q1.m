close all;

maze = importdata('Q1_map.txt', ' '); % 0: wall, 1: road, 2: start, 3: goal
R = importdata('Q1_reward.txt', ' ');
if size(maze) ~= size(R)
    error('Size of maze and rewards should be the same.');
end
pi = zeros(size(maze, 1), size(maze, 2), 4) + 0.25; % pi(s, a) = 1/4 for all s, a
A = [-1, 0; 0, 1; 1, 0; 0, -1]; % [N; E; S; W]
para.stop_thres = 0.001; % stop threshold
para.discount_factor = 0.9; % discount factor
para.prob_follow_act = 0.7;

hold all;
P = get_P(maze, A, para);
V_Q1A = policy_evaluation(maze, A, P, R, pi, para);
disp('Q1 (A) Finished. See V_Q1A for the resulting values.');
para.discount_factor = 0.7; % discount factor
V_Q1B = policy_evaluation(maze, A, P, R, pi, para);
legend('Q1 A: gamma = 0.9', 'Q1 B: gamma = 0.7');
hold off;
disp('Q1 (B) Finished. See V_Q1B for the resulting values.');