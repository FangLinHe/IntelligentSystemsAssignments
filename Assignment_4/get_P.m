function P = get_P(maze, A, para)
P = zeros(size(maze, 1), size(maze, 2), 4, size(maze, 1), size(maze, 2));

for i = 1:size(maze, 1)
    for j = 1:size(maze, 2)
        if maze(i ,j) % value = 1: not wall
            for a = 1:size(A, 1) % expected action
                for ra = 1:size(A, 1) % real action
                    nst.y = i + A(ra, 1);
                    nst.x = j + A(ra, 2);
                    if nst.y == 0
                        nst.y = size(maze, 1);
                    elseif nst.y == size(maze, 1)
                        nst.y = 1;
                    end
                    if nst.x == 0
                        nst.x = size(maze, 2);
                    elseif nst.y == size(maze, 2)
                        nst.x = 1;
                    end
                    if maze(i, j) == 3 || maze(nst.y, nst.x) == 0 % already at goal state or the next state is wall
                        nst.y = i;
                        nst.x = j;
                    end
                    prob = para.prob_follow_act * (a == ra) + ((1-para.prob_follow_act)/(size(A,1)-1)) * (a ~= ra);
                    P(i, j, a, nst.y, nst.x) = P(i, j, a, nst.y, nst.x) + prob;
                end
            end
        end
    end
end
