num_exp = 4;
chromosome_size = [5; 10; 20; 50];
%num_offsprings = chromosome_size * 10;
num_offsprings = [80; 80; 170; 340];
learning_rate = 1./sqrt(chromosome_size);
iteration_time = 100;
repeat_time = 10;
draw = false;

close all;
fitness = zeros(num_exp, iteration_time, repeat_time);
for i = 1:num_exp
    for j = 1:repeat_time
        fprintf('.');
        fitness(i, :, j) = EvolutionStrategy(chromosome_size(i), num_offsprings(i), learning_rate(i), iteration_time, draw);
    end
    fprintf('\n');
    
    meanfitness = mean(fitness(i,:,:), 3);
    hold all;
    %plot(log10(meanfitness));
    plot(meanfitness);
    drawnow;
end
legend('5', '10', '20', '50');