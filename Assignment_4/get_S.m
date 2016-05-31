function S = get_S(maze)
S = zeros(numel(maze), 2);
for i = 1:size(maze, 1)
    ss = (i-1)*size(maze, 2) + 1;
    se = i*size(maze, 2);
    S(ss:se, 1) = i;
    S(ss:se, 2) = (1:size(maze, 2))';
end