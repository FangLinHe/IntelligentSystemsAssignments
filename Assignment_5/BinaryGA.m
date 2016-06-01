function [bestfitness, stop_crit] = BinaryGA(chromosome_size, population_size, mutation_rate, iteration_time, good_solution_percent)
draw = false;
if nargin < 1
    chromosome_size = 5;
    population_size = 20;
    mutation_rate = 0.005;
    iteration_time = 100;
    good_solution_percent = 1;
end
population_size = population_size - mod(population_size, 2);

% 1. Initialize random population of candidate solutions
population = InitializeGA(chromosome_size, population_size);
if prod(sum(cell2mat(population))) == 0
    warning('More than one digit in all populations are all zeros. \nMay take more time to find a good solution.');
end

bestfitness = zeros(iteration_time, 1);
if draw
    close all;
    figure;
    allfitness = zeros(population_size, 1);
end
count = 0;
stop_crit = 0;
while count < iteration_time %&& ~stop_crit
    % 5. Deterimine if the solution is good
    if  stop_crit == 0 && IsGoodSolution(population, 'part', good_solution_percent)
        stop_crit = count;
    end
    %{
    if stop_crit
        break;
    end
    %}
        
    % 2. Evaluate solutions on problem and assign a fitness score
    fitness = GetFitness(population);

    % 3. Select some solutions for mating
    parents_index = SelectParents(fitness);

    % 4. Recombine create new solutions from selected ones by exchanging structure
    offsprings = Crossover(population, parents_index);
    population = Mutation(offsprings, mutation_rate);
    
    count = count + 1;
    bestfitness(count) = max(fitness);
    if draw
        allfitness(count) = sum(sum(cell2mat(population)));
        plot(allfitness(1:count));
        drawnow;
    end
end

%--------------------------------------------------------------------------

function population = InitializeGA(chromosome_size, population_size)
population = cell(population_size, 1);
for i = 1:population_size
    population{i} = (rand(1, chromosome_size) > 0.5);
end

function fitness = GetFitness(population)
fitness = zeros(numel(population), 1);
for i = 1:numel(population)
    fitness(i) = sum(population{i});
end
fitness = fitness ./ sum(fitness);

function stop_crit = IsGoodSolution(population, para, good_solution_percent)
if nargin < 2
    para = 'part'; % 'all': all the populations are good solutions; % 'part': over a half of the population are good solutions
end
if nargin < 3
    good_solution_percent = 0.5;
end
if strcmpi(para, 'all')
    if sum(sum(cell2mat(population))) == numel(population) * numel(population{1})
        stop_crit = true;
    else
        stop_crit = false;
    end
else
    sum_population = sum(cell2mat(population), 2);
    if sum(sum_population == numel(population{1})) >= numel(population) * good_solution_percent;
        stop_crit = true;
    else
        stop_crit = false;
    end
end
    

function parents_index = SelectParents(fitness)
cumsum_fitness = cumsum(fitness);
parents_index = zeros(numel(fitness), 1);
for i = 1:numel(fitness)
    r = rand;
    parents_index(i) = find(cumsum_fitness>=r,1);
end

function offsprings = Crossover(population, parents_index)
offsprings = cell(numel(population), 1);
population_size = numel(population{1});
for i = 1:2:numel(population)
    crossover_point = randi(population_size+1, 1) - 1;
    offsprings{i} = [population{parents_index(i)}(1, 1:crossover_point), population{parents_index(i+1)}(1, crossover_point+1:end)];
    offsprings{i+1} = [population{parents_index(i+1)}(1, 1:crossover_point), population{parents_index(i)}(1, crossover_point+1:end)];
end

function offsprings = Mutation(offsprings, mutation_rate)
for i = 1:numel(offsprings)
    for j = 1:numel(offsprings{1})
        r = rand;
        if r < mutation_rate
            offsprings{i}(j) = ~offsprings{i}(j);
        end
    end
end