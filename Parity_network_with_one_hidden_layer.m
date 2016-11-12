%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set up
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dataset = generate_data(4);
targets = generate_targets(dataset);

m = size(dataset, 1);

% add bias term to the inputs
dataset = [ones(m, 1), dataset];

% initiate weights
% w1 - 5x4, weights connecting input layer to the hidden layer
% w2 - 5x1, weights connecting the hidden layer to the output layer
w1 = rand(5, 4) - 0.5;
w2 = rand(5, 1) - 0.5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% performing the actual calculations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[a1, a2, z1, z2] = feed_forward(w1, w2, dataset, targets);
fprintf('Misclassified example count before training: %d\n\n', ...
        error_count(w1, w2, dataset, targets))

i = 0;
while true
  [a1, a2, z1, z2] = feed_forward(w1, w2, dataset, targets);
  [g1, g2] = calculate_gradient(w1, w2, dataset, targets);
  w1 = w1 - g1;
  w2 = w2 - g2;
  i = i + 1;
  if mod(i, 1000) == 0 & error_count(w1, w2, dataset, targets) == 0
    break
  end
end

fprintf('Training completed after %d iterations\n', i);
fprintf('Misclassified example count after training: %d\n\n', ...
        error_count(w1, w2, dataset, targets))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% definitions of functions used in the calculations above
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [a1, a2, z1, z2] = feed_forward(w1, w2, dataset, targets)
  m = size(dataset, 1);

  z1 = dataset * w1;
  a1 = sigmoid(z1);

  % add the bias unit to the activations of the hidden layer
  a1 = [ones(m, 1), a1];

  z2 = a1 * w2;
  a2 = sigmoid(z2);
end

function [g1, g2] = calculate_gradient(w1, w2, dataset, targets)
  m = size(dataset, 1);
  [a1, a2, z1, z2] = feed_forward(w1, w2, dataset, targets);

  delta2 = a2 - targets;
  g2 = zeros(size(w2));
  for i = 1:size(a2,1)
    g2 = g2 + delta2(i) * a1(i, :)';
  end

  delta1 = delta2 * w2(2:end)' .* a1(:, 2:end) .* (1 - a1(:, 2:end));
  g1 = zeros(size(w1));
  for i = 1:size(dataset, 1)
    g1 = g1 + dataset(i, :)' * delta1(i, :);
  end

  g2 = g2 / m;
  g1 = g1 / m;
end

function [g1, g2] = calculate_numerical_gradient(w1, w2, dataset, targets)
  % calculate gradient with respect to weights in w1 and w2
  delta = 1e-4;
  g1 = zeros(size(w1));
  g2 = zeros(size(w2));

  for i = 1:size(w1, 1)
    for j = 1:size(w1, 2)
      old_w1 = w1;
      w1(i,j) = w1(i, j) - delta;
      [~, a2, ~, ~] = feed_forward(w1, w2, dataset, targets);
      cost_1 = calculate_cost(a2, targets);

      w1 = old_w1;
      w1(i,j) = w1(i, j) + delta;
      [~, a2, ~, ~] = feed_forward(w1, w2, dataset, targets);
      cost_2 = calculate_cost(a2, targets);

      g1(i, j) = (cost_2 - cost_1) / (2 * delta);
      w1 = old_w1;
    end
  end

  for i = 1:size(w2, 1)
    for j = 1:size(w2, 2)
      old_w2 = w2;
      w2(i,j) = w2(i, j) - delta;
      [~, a2, ~, ~] = feed_forward(w1, w2, dataset, targets);
      cost_1 = calculate_cost(a2, targets);

      w2 = old_w2;
      w2(i,j) = w2(i, j) + delta;
      [~, a2, ~, ~] = feed_forward(w1, w2, dataset, targets);
      cost_2 = calculate_cost(a2, targets);

      g2(i, j) = (cost_2 - cost_1) / (2 * delta);
      w2 = old_w2;
    end
  end
end

function count = error_count(w1, w2, dataset, targets)
  count = sum(predictions(w1, w2, dataset, targets) ~= targets);
end

function y = predictions(w1, w2, dataset, targets)
  [~, a2, ~, ~] = feed_forward(w1, w2, dataset, targets);
  y = a2 > 0.5;
end

function c = calculate_cost(h, t)
  % calculate the cross-entropy cost
  m = size(t, 1);
  c = (-1/m) * (sum(t' * log(h) + (1 - t)' * log(1 - h)));
end

function s = sigmoid(z)
  s = 1 ./ (1 + exp(-z));
end

function dataset = generate_data(n, dataset)
  % recursively generates the dataset which contains 2^n examples of length n

  if nargin == 1
    dataset = [0; 1];
    n = n - 1;
  end

  if n == 0
    return
  end

  m = size(dataset, 1);

  dataset = [dataset, zeros(m, 1);
             dataset, ones(size(dataset, 1), 1);];

  dataset = generate_data(n - 1, dataset);
end

function t = generate_targets(dataset)
  summed_rows = sum(dataset, 2);
  t = mod(summed_rows, 2);
end
