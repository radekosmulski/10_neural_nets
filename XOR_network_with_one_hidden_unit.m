%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set up
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% creating the data for a XOR function - already with the bias unit in the first
% column

X = [
  1, 0, 0;
  1, 0, 1;
  1, 1, 0;
  1, 1, 1;
];

% target vector of correct values
t = [0; 1; 1; 0];

% initial weights for input neurons and the middle neuron feeding into
% the output unit
theta_out = rand(4, 1);
% initial weights for connections from input neurons to the neuron in the middle
theta_mid = rand(3, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% performing the actual calculations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Count of incorrectly classified examples before training: %d\n', ...
        get_error_count(theta_out, theta_mid, t, X))

costs = [];
while isempty(costs) || costs(end) > 0.1
  costs(end + 1) = get_cost(theta_out, theta_mid, t, X);
  [grad_out, grad_mid] = get_grad(theta_out, theta_mid, t, X);
  theta_out = theta_out - grad_out;
  theta_mid = theta_mid - grad_mid;
end

% Plot costs during training and provide summary information
plot(costs);
title('Cost vs iteration');
ylabel('cost');
xlabel('iteration #');
fprintf('Training completed after %d iterations\n', size(costs, 2));
fprintf('Count of incorrectly classified examples after training: %d\n', ...
        get_error_count(theta_out, theta_mid, t, X))
fprintf('\n <press any key to continue>\n')
pause;
close;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% definitions of functions used in the calculations above
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [z_mid, z_out, a_mid, a_out] = forward_prop(theta_out, theta_mid, X)
  z_mid = X * theta_mid;
  a_mid = [X, sigmoid(z_mid)];
  z_out = a_mid * theta_out;
  a_out = sigmoid(z_out);
end

function h = sigmoid(z)
  h = 1 ./ (1 + exp(-z));
end

function cost = get_cost(theta_out, theta_mid, t, X)
  [~, ~, ~, a_out] = forward_prop(theta_out, theta_mid, X);
  cost = 0.5 * (t - a_out)' * (t - a_out);
end

function count = get_error_count(theta_out, theta_mid, t, X)
  [~, ~, ~, a_out] = forward_prop(theta_out, theta_mid, X);
  count = sum((a_out > 0.5) ~= t);
end

function [grad_out, grad_mid] = get_grad(theta_out, theta_mid, t, X)
  grad_out = zeros(size(theta_out));
  grad_mid = zeros(size(theta_mid));

  [~, ~, a_mid, a_out] = forward_prop(theta_out, theta_mid, X);
  delta_out = (t - a_out) .* a_out .* (1 - a_out);
  delta_mid = delta_out * theta_out(4) .* a_mid(:, 4) .* (1 - a_mid(:, 4));

  for i=1:4
    grad_mid = grad_mid - X(i, :)' * delta_mid(i);
    grad_out = grad_out - a_mid(i, :)' * delta_out(i);
  end
end

function [grad_out, grad_mid] = get_numerical_grad(theta_out, theta_mid, t, X)
  delta = 1e-4;

  grad_out = zeros(size(theta_out));
  for i = 1:size(theta_out, 1)
    for j = 1:size(theta_out, 2)
      old_val = theta_out(i, j);

      theta_out(i, j) = old_val - delta;
      cost_a = get_cost(theta_out, theta_mid, t, X);
      theta_out(i, j) = old_val + delta;
      cost_b = get_cost(theta_out, theta_mid, t, X);
      grad_out(i, j) = (cost_b - cost_a) / (2 * delta);

      theta_out(i, j) = old_val;
    end
  end

  grad_mid = zeros(size(theta_mid));
  for i = 1:size(theta_mid, 1)
    for j = 1:size(theta_mid, 2)
      old_val = theta_mid(i, j);

      theta_mid(i, j) = old_val - delta;
      cost_a = get_cost(theta_out, theta_mid, t, X);
      theta_mid(i, j) = old_val + delta;
      cost_b = get_cost(theta_out, theta_mid, t, X);
      grad_mid(i, j) = (cost_b - cost_a) / (2 * delta);

      theta_mid(i, j) = old_val;
    end
  end
end
