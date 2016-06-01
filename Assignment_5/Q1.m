num_exp = 4;
chromosome_size = [5; 10; 20; 50];
population_size = [80; 80; 170; 340];
mutation_rate = 0.00001;
iteration_time = 350;
good_solution_percent = 0.75;
repeat_time = 10;

close all;
figure;
bestfitness = zeros(num_exp, iteration_time, repeat_time);
for i = 1:num_exp
    for j = 1:repeat_time
        fprintf('.');
        [bestfitness(i, :, j), get_stop_crit] = BinaryGA(chromosome_size(i), population_size(i), mutation_rate, iteration_time, good_solution_percent);
        if get_stop_crit == 0
            warning('The program has not reached the optimal solution criteria.');
        end
    end
    fprintf('\n');
    meanfitness = mean(bestfitness(i,:,:), 3);
    hold all;
    plot(meanfitness - min(min(bestfitness(i,:,:))));
    drawnow;
end
legend('5', '10', '20', '50');