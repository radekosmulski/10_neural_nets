%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set up
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% create data
dataset = [
  1 0 0;
  1 1 0;
  1 1 1;
];

% generate targets
targets = [
  1 0 0;
  0 1 0;
  0 0 1;
];

% add the bias unit to inputs
dataset = [ones(3, 1), dataset];

% This will be a very simple NN, consisting only of the input layer and
% the softmax group. As such, it will only have one set of weights.
weights = rand(4, 3) - 0.5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Performing the actual calculations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

i = 0;
fprintf('Misclassified example count before training: %d\n\n', ...
        error_count(weights, dataset, targets))
while true
  weights = weights - gradient(weights, dataset, targets);
  i = i + 1;
  if error_count(weights, dataset, targets) == 0
    break
  end
end
fprintf('Training completed in %d iterations\n', i)
fprintf('Misclassified example count after training: %d\n\n', ...
        error_count(weights, dataset, targets))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definitions of functions used in the calculations above
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [probabilities] = feed_forward(weights, dataset, targets)
  z = dataset * weights;
  numerator = exp(z);
  denumerator = sum(numerator, 2);
  probabilities = numerator ./ denumerator;
end

function [grad] = numerical_gradient(weights, dataset, targets)
  grad = zeros(size(weights));
  delta = 1e-4;

  for i = 1:size(weights, 1)
    for j = 1:size(weights, 2)
      old_v = weights(i, j);

      weights(i, j) = weights(i, j) - delta;
      cost_a = cost(weights, dataset, targets);
      weights(i, j) = old_v;

      weights(i, j) = weights(i, j) + delta;
      cost_b = cost(weights, dataset, targets);
      weights(i, j) = old_v;

      grad(i, j) = (cost_b - cost_a) / (2 * delta);
    end
  end
end

function [grad] = gradient(weights, dataset, targets)
  probabilities = feed_forward(weights, dataset, targets);
  grad = zeros(size(weights));
  for i = 1:size(dataset, 1)
    grad = grad + dataset(i, :)' * (targets(i, :) - probabilities(i, :));
  end
  grad = -grad;
end

function [h] = hypothesis(weights, dataset, targets)
  [vals, idx] = max(feed_forward(weights, dataset, targets), [], 2);
  h = zeros(3,3);
  for i = 1:3
    h(i, idx(i)) = 1;
  end
end

function [count] = error_count(weights, dataset, targets)
  h = hypothesis(weights, dataset, targets);
  misclassified_examples = ~all(h == targets, 2);
  count = sum(misclassified_examples);
end

function [c] = cost(weights, dataset, targets)
  h = hypothesis(weights, dataset, targets);
  probabilities = feed_forward(weights, dataset, targets);
  % normally the below will not work
  individual_costs = -log(probabilities) .* targets;
  c = sum(sum(individual_costs));
end
