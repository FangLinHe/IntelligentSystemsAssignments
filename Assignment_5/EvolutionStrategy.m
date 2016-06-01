function out = EvolutionStrategy(chromosome_size, num_offsprings, learning_rate, iteration_time, draw)

if nargin < 1
    chromosome_size = 5;
    num_offsprings = 40;
    learning_rate = 1/sqrt(chromosome_size);
    iteration_time = 300;
    draw = false;
end

% 1. Initialize parents and evaluate them
chromosome = InitializeES(chromosome_size);
value = Rosenbrock(chromosome(1:end-1));

if draw
    figure;
    all_fitness = zeros(iteration_time, 1);
end

count = 0;
while count < iteration_time %&& ~stop_crit
    count = count + 1;
        
    % 2. Create some offspring by perturbing parents with Gaussian noise 
    %    according to parent's mutation parameters
    offsprings = CreateOffsprings(chromosome, num_offsprings, learning_rate);
    
    % 3. Evaluate offspring: refer to the function CreateOffsprings
    % 4. Select new parents from offspring and possibly old parents
    [new_parent, all_fitness(count)] = SelectNewParents(chromosome, offsprings);
    chromosome = new_parent;
    if draw
        plot(all_fitness);
        drawnow;
    end
end
out = all_fitness;

%--------------------------------------------------------------------------

function population = InitializeES(chromosome_size)
population = rand(1, chromosome_size+1) * 15 - 5;

function sum = Rosenbrock(x)
sum = 0;
for i = 1:numel(x)-1
    sum = sum + (1-x(i))^2 + 100 * (x(i+1) - (x(i))^2)^2;
end

function offsprings = CreateOffsprings(chromosome, num_offsprings, learning_rate)
if nargin<3
    learning_rate = 0.1;
end
offsprings = cell(num_offsprings, 1);
for i = 1:num_offsprings
    offsprings{i} = chromosome;
    offsprings{i}(end) = offsprings{i}(end) * exp(normrnd(0, learning_rate));
    offsprings{i}(1, 1:end-1) = offsprings{i}(1, 1:end-1) + normrnd(0, learning_rate, 1, numel(chromosome)-1);
end

function [new_parent, best_value, values] = SelectNewParents(parents, offsprings)
all_chromosomes = [parents; offsprings];
values = zeros(numel(all_chromosomes), 1);
for i = 1:numel(all_chromosomes)
    values(i) = Rosenbrock(all_chromosomes{i}(1:end-1));
end
[best_value, ind] = min(values);
new_parent = all_chromosomes{ind};