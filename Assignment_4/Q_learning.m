function [Q, acum_reward, exe_time_record] = Q_learning(maze, A, R, para)
if nargin < 4
    error('Too few parameters');
end

episode = 1;
Q = zeros(size(maze, 1), size(maze, 2), size(A, 1)); % initialize the Q values
fileID = fopen(para.path_filename, 'w+'); % open the file to write the path
exe_time_record = zeros(para.max_episode, 1);
acum_reward = zeros(para.max_episode, 1); % accumulated rewards
while episode <= para.max_episode % for each episode
    fprintf(fileID, '--------------------------\r\n%d:\r\n', episode); % write file: current episode
    [s(1), s(2)] = find(maze == 2); % initialize s = find the start point: 2 in the maze
    [g(1), g(2)] = find(maze == 3); % find the goal point: 3 in the maze
    exe_time = 1;
    while exe_time <= para.max_actions && sum(s ~= g)
        a = choose_a(maze, Q, s, A, para);
        [r, s2] = observe(maze, R, s, A, a, para);
        Q(s(1), s(2), a) = Q(s(1), s(2), a) + ...
            para.learning_rate * (r + para.discount_factor * ...
            max(Q(s2(1), s2(2), :)) - Q(s(1), s(2), a));
        s = s2;
        fprintf(fileID, '%d\t%d %d\r\n', exe_time, s2(1), s2(2));
        exe_time = exe_time + 1;
    end
    acum_reward(episode) = acum_reward(max(1,episode-1)) + prod(s == g) * max(R(:));
    exe_time_record(episode) = exe_time;
    disp(exe_time-1);
    episode = episode + 1;
end

function a = choose_a(maze, Q, s, A, para)
if sum(Q(s(1), s(2), :)) == 0
    max_ind = randi(4, 1);
else
    [~, max_ind] = max(Q(s(1), s(2), :));
end
a_other = 1:size(A, 1);
if max_ind ~= size(A, 1)
    a_other(max_ind:end-1) = a_other(max_ind+1:end);
end
a_other = a_other(1:end-1);

if rand >= para.epsilon
    a = max_ind;
else
    a = a_other(randi(size(A, 1)-1, 1));
end

function [r, s2] = observe(maze, R, s, A, a, para)
rand_num = rand;
prob = zeros(size(A, 1)+1, 1) + (1-para.prob_follow_act) / (size(A,1)-1);
prob(a) = para.prob_follow_act;
prob = cumsum([0; prob(1:end-1)]);
ra = find(prob>rand_num, 1) - 1; % real action
s2 = s + A(ra, :);
if s2(1) <= 0
    s2(1) = size(maze, 1) + s2(1);
end
if s2(2) <= 0
    s2(2) = size(maze, 2) + s2(2);
end
if s2(1) > size(maze, 1)
    s2(1) = s2(1) - size(maze, 1);
end
if s2(2) > size(maze, 2)
    s2(2) = s2(2) - size(maze, 2);
end
    
if maze(s2(1), s2(2)) == 0
    s2 = s;
end
r = R(s2(1), s2(2));
    