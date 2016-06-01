training_label = zeros(6000, 10);
test_label = zeros(1000, 10);

for i = 1:6000
    training_label(i, training_label_temp(i)+1) = 1;
end

for i = 1:1000
    test_label(i, test_label_temp(i)+1) = 1;
end

clear training_label_temp test_label_temp i;