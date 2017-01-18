from sklearn.datasets import fetch_mldata
from scipy.ndimage import zoom
import numpy as np


notice = '''
*********************************************************************************
The training of these NNs is quite slow as I have not implemented any of
the methods to help with convergence and on top of that given the size of
the dataset and similiarity of examples minibatch gradient descent would likely
be a better approach.

Once fully trained, the benchmark NN (having the same architecture as an expert)
achieves a classification error of around 6.5% on the test set.

The mixture of experts tops this result and achieves a classification error of
around 3.5% on the test set.

Usually a single expert is chosen as the lead on all the examples, though it is
not always the case. It would be interesting to see what results could be
achieved with pretraining of experts and pretraining of the gating NN
(for instance, each expert could be pretrained on a different pair of digits to
further promote specialization). Though probably given the subpar performance of
the squared error cost function in the context of mixture of experts (in the
beginning the expert making the greatest mistake learns the most) one could get
more mileage from simply swapping out the cost function for a better one.

Assuming you let the script run for long enough, you should achieve results
in roughly similar ballpark to the ones mentioned above.
*********************************************************************************
'''

print(notice)

################################
# Prepare train and test sets
################################

print('Fetching MNIST dataset. Please wait...\n')
dataset = fetch_mldata('MNIST original', data_home='datasets')

print('Resizing the images to make them smaller\n')
resized_dataset = np.zeros((dataset.data.shape[0], 14, 14))
for i in range(dataset.data.shape[0]):
    resized_dataset[i] = zoom(dataset.data[i].reshape((28,28)), 0.5, order = 3)
resized_dataset = resized_dataset.reshape((-1, 14 * 14))

# Normalization to help with training in a network with a sigmoid nonlinearity
resized_dataset += resized_dataset.min()
resized_dataset /= resized_dataset.max()

# Shuffling the ids to prepare for creation of train and test sets
ids = np.arange(len(dataset.data))
np.random.shuffle(ids)

# The full dataset consists of 70000 labelled examples.
# We will use 60000 examples for training and 10000 for our test set.
n_rows_train = 60000
n_rows_test = len(dataset.target) - n_rows_train
data_train = np.c_[np.ones((n_rows_train, 1)), resized_dataset[ids[:n_rows_train], :]]
targets_train = np.zeros((n_rows_train, 10))
targets_train[np.arange(n_rows_train), dataset.target[ids[:n_rows_train]].astype(int)] = 1
data_test = np.c_[np.ones((n_rows_test, 1)), resized_dataset[ids[n_rows_train:], :]]
targets_test = np.zeros((n_rows_test, 10))
targets_test[np.arange(n_rows_test), dataset.target[ids[n_rows_train:]].astype(int)] = 1

################################
# Function definitions
################################

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def feed_forward(l2_weights, l3_weights, data):
    z2 = data.dot(l2_weights)
    a2 = np.c_[np.ones((z2.shape[0], 1)), sigmoid(z2)]
    z3 = a2.dot(l3_weights)
    a3 = sigmoid(z3)
    return a2, a3

def mean_squared_error(hypotheses, targets):
    return np.sum(0.5 * (hypotheses - targets) ** 2) / hypotheses.shape[0]

def mean_squared_error_signal(hypotheses, targets):
    return hypotheses - targets

def classification_error(hypotheses, targets):
    return 100 - np.mean(hypotheses.argmax(1) == targets.argmax(1)).round(4) * 100

def gradient_expert_NN(signal, a2, a3, l2_weights, l3_weights, data):
    a3_delta = signal * a3 * (1 - a3)
    l3_weights_grad = a2.T.dot(a3_delta) / data.shape[0]
    a2_grad = a3_delta.dot(l3_weights[1:, :].T)
    a2_delta = a2_grad * a2[:, 1:] * (1 - a2[:, 1:])
    l2_weights_grad = data.T.dot(a2_delta) / data.shape[0]
    return l2_weights_grad, l3_weights_grad

def numerical_gradient_expert_NN(l2_weights, l3_weights, data, targets):
    delta = 1e-4

    l2_weights_grad = np.zeros((l2_weights.shape))
    for i in range(l2_weights.shape[0]):
        for j in range(l2_weights.shape[1]):
            l2_weights[i][j] -= delta
            cost_minus = mean_squared_error(feed_forward(l2_weights, l3_weights, data)[-1], targets)
            l2_weights[i][j] += 2 * delta
            cost_plus = mean_squared_error(feed_forward(l2_weights, l3_weights, data)[-1], targets)
            l2_weights_grad[i][j] = (cost_plus - cost_minus) / (2 * delta)
            l2_weights[i][j] -= delta

    l3_weights_grad = np.zeros((l3_weights.shape))
    for i in range(l3_weights.shape[0]):
        for j in range(l3_weights.shape[1]):
            l3_weights[i][j] -= delta
            cost_minus = mean_squared_error(feed_forward(l2_weights, l3_weights, data)[-1], targets)
            l3_weights[i][j] += 2 * delta
            cost_plus = mean_squared_error(feed_forward(l2_weights, l3_weights, data)[-1], targets)
            l3_weights_grad[i][j] = (cost_plus - cost_minus) / (2 * delta)
            l3_weights[i][j] -= delta

    return l2_weights_grad, l3_weights_grad

def train_benchmark_NN(epoch_count = 1, learning_rate = 0.1):
    global benchmark_l2_weights, benchmark_l3_weights
    print('Training benchmark NN for', epoch_count, 'epoch(s) with learning rate', learning_rate)

    _, hypotheses = feed_forward(benchmark_l2_weights, benchmark_l3_weights, data_test)
    print('Test set cost before training:', mean_squared_error(hypotheses, targets_test))
    print('Test set classification error before training: %0.2f%%' % classification_error(hypotheses, targets_test))

    for e in range(epoch_count):
        a2, hypotheses = feed_forward(benchmark_l2_weights, benchmark_l3_weights, data_train)
        signal = mean_squared_error_signal(hypotheses, targets_train)
        l2_grad, l3_grad = gradient_expert_NN(signal, a2, hypotheses, benchmark_l2_weights, benchmark_l3_weights, data_train)
        benchmark_l2_weights -= learning_rate * l2_grad
        benchmark_l3_weights -= learning_rate * l3_grad

    _, hypotheses = feed_forward(benchmark_l2_weights, benchmark_l3_weights, data_test)
    print('Test set cost after training:', mean_squared_error(hypotheses, targets_test))
    print('Test set classification error after training: %0.2f%%\n' % classification_error(hypotheses, targets_test))

################################
# Train a benchmark NN
################################

# Implementing a NN just like the NNs that will form the mixture of experts.
# This will give me a reference point for benchmarking whether the mixture of
# experts improves on those results and by how much.

benchmark_l2_weights = np.random.random((197, 25)) - 0.5
benchmark_l3_weights = np.random.random((26, 10)) - 0.5

print('Training benchmark NN, might take about 10 minutes depending on your hardware...\n')
train_benchmark_NN(2000, 10)

################################
# Function definitions specific to mixture of experts
################################

def feed_forward_gating_NN(l2_weights, softmax_weights, data):
    z2 = data.dot(l2_weights)
    a2 = np.c_[np.ones((z2.shape[0], 1)), sigmoid(z2)]
    z3 = a2.dot(softmax_weights)

    # calculating softmax activations
    z3 -= z3.max() # avoiding overflow
    z3_exp = np.exp(z3)
    z3_row_sum = z3_exp.sum(1)
    a3 = z3_exp / z3_row_sum.reshape((-1,1))

    return a2, a3

def feed_forward_ME(data):
    global gating_l2_weights, gating_softmax_weights, expert_l2_weights, expert_l3_weights

    gating_l2_states, gating_hypotheses = feed_forward_gating_NN(gating_l2_weights, gating_softmax_weights, data)

    expert_l2_states = np.zeros((10, data.shape[0], 26))
    expert_hypotheses = np.zeros((10, data.shape[0], 10))
    for i in range(10):
        l2_states, hypotheses = feed_forward(expert_l2_weights[i], expert_l3_weights[i], data)
        expert_l2_states[i] = l2_states
        expert_hypotheses[i] = hypotheses
    return gating_l2_states, gating_hypotheses, expert_l2_states, expert_hypotheses

def numerical_gradient_ME(data, targets):
    global gating_l2_weights, gating_softmax_weights, expert_l2_weights, expert_l3_weights
    delta = 1e-4

    gating_l2_weights_grad = np.zeros((gating_l2_weights.shape))
    for i in range(gating_l2_weights.shape[0]):
        for j in range(gating_l2_weights.shape[1]):
            gating_l2_weights[i][j] -= delta
            _, gating_hypotheses, _, expert_hypotheses = feed_forward_ME(data)
            cost_minus = mean_squared_error_ME(gating_hypotheses, expert_hypotheses, targets)
            gating_l2_weights[i][j] += 2 * delta
            _, gating_hypotheses, _, expert_hypotheses = feed_forward_ME(data)
            cost_plus = mean_squared_error_ME(gating_hypotheses, expert_hypotheses, targets)
            gating_l2_weights_grad[i][j] = (cost_plus - cost_minus) / (2 * delta)
            gating_l2_weights[i][j] -= delta

    gating_softmax_weights_grad = np.zeros((gating_softmax_weights.shape))
    for i in range(gating_softmax_weights.shape[0]):
        for j in range(gating_softmax_weights.shape[1]):
            gating_softmax_weights[i][j] -= delta
            _, gating_hypotheses, _, expert_hypotheses = feed_forward_ME(data)
            cost_minus = mean_squared_error_ME(gating_hypotheses, expert_hypotheses, targets)
            gating_softmax_weights[i][j] += 2 * delta
            _, gating_hypotheses, _, expert_hypotheses = feed_forward_ME(data)
            cost_plus = mean_squared_error_ME(gating_hypotheses, expert_hypotheses, targets)
            gating_softmax_weights_grad[i][j] = (cost_plus - cost_minus) / (2 * delta)
            gating_softmax_weights[i][j] -= delta

    expert_l2_weights_grad = np.zeros((expert_l2_weights.shape))
    for e in range(expert_l2_weights.shape[0]):
        for i in range(expert_l2_weights.shape[1]):
            for j in range(expert_l2_weights.shape[2]):
                expert_l2_weights[e][i][j] -= delta
                _, gating_hypotheses, _, expert_hypotheses = feed_forward_ME(data)
                cost_minus = mean_squared_error_ME(gating_hypotheses, expert_hypotheses, targets)
                expert_l2_weights[e][i][j] += 2 * delta
                _, gating_hypotheses, _, expert_hypotheses = feed_forward_ME(data)
                cost_plus = mean_squared_error_ME(gating_hypotheses, expert_hypotheses, targets)
                expert_l2_weights_grad[e][i][j] = (cost_plus - cost_minus) / (2 * delta)
                expert_l2_weights[e][i][j] -= delta

    expert_l3_weights_grad = np.zeros((expert_l3_weights.shape))
    for e in range(expert_l3_weights.shape[0]):
        for i in range(expert_l3_weights.shape[1]):
            for j in range(expert_l3_weights.shape[2]):
                expert_l3_weights[e][i][j] -= delta
                _, gating_hypotheses, _, expert_hypotheses = feed_forward_ME(data)
                cost_minus = mean_squared_error_ME(gating_hypotheses, expert_hypotheses, targets)
                expert_l3_weights[e][i][j] += 2 * delta
                _, gating_hypotheses, _, expert_hypotheses = feed_forward_ME(data)
                cost_plus = mean_squared_error_ME(gating_hypotheses, expert_hypotheses, targets)
                expert_l3_weights_grad[e][i][j] = (cost_plus - cost_minus) / (2 * delta)
                expert_l3_weights[e][i][j] -= delta

    return expert_l2_weights_grad, expert_l3_weights_grad, gating_l2_weights_grad, gating_softmax_weights_grad

def error_of_each_expert_by_example(expert_hypotheses, targets):
    individual_errors = np.zeros(expert_hypotheses[0].shape)
    for i in range(10):
        individual_errors[:, i] = np.sum((expert_hypotheses[i] - targets) ** 2, 1)
    return individual_errors

def mean_squared_error_ME(gating_hypotheses, expert_hypotheses, targets):
    weighted_individual_errors = gating_hypotheses * error_of_each_expert_by_example(expert_hypotheses, targets)
    return 0.5 * np.sum(weighted_individual_errors) / targets.shape[0]

def hypothesis_ME(gating_hypotheses, expert_hypotheses):
    hypothesis_ME = np.zeros((expert_hypotheses[0].shape))
    for i in range(expert_hypotheses.shape[0]):
        hypothesis_ME += gating_hypotheses[:, i].reshape((-1,1)) * expert_hypotheses[i]
    return hypothesis_ME

def gradient_ME(gating_l2_layer_states, gating_hypotheses, expert_l2_states, expert_hypotheses, data, targets):
    individual_errors = error_of_each_expert_by_example(expert_hypotheses, targets)
    collective_errors = np.sum(gating_hypotheses * individual_errors, 1)

    delta_softmax = gating_hypotheses * (individual_errors - collective_errors.reshape((-1,1)))
    gating_softmax_weights_grad = gating_l2_layer_states.T.dot(delta_softmax) / targets.shape[0]

    delta_gating_l2 = delta_softmax.dot(gating_softmax_weights[1:, :].T)
    gating_l2_grad = delta_gating_l2 * gating_l2_layer_states[:, 1:] * (1 - gating_l2_layer_states[:, 1:])
    gating_l2_weights_grad = data.T.dot(gating_l2_grad) / targets.shape[0]

    expert_l2_weights_grad = np.zeros(expert_l2_weights.shape)
    expert_l3_weights_grad = np.zeros(expert_l3_weights.shape)
    for i in range(10):
        signal = gating_hypotheses[:, i].reshape((-1,1)) * (expert_hypotheses[i] - targets)
        expert_l2_weights_grad[i], expert_l3_weights_grad[i] = gradient_expert_NN(signal, expert_l2_states[i], expert_hypotheses[i], expert_l2_weights[i], expert_l3_weights[i], data)

    return gating_l2_weights_grad, gating_softmax_weights_grad, expert_l2_weights_grad, expert_l3_weights_grad

def train_mixture_of_experts(epoch_count = 1, learning_rate = 0.1):
    global gating_l2_weights, gating_softmax_weights, expert_l2_weights, expert_l3_weights
    print('Training mixture of experts for', epoch_count, 'epoch(s) with learning rate', learning_rate)

    _, gating_hypotheses, _, expert_hypotheses = feed_forward_ME(data_test)
    print('Test set cost before training:', mean_squared_error_ME(gating_hypotheses, expert_hypotheses, targets_test))
    print('Test set classification error before training: %0.2f%%' % classification_error(hypothesis_ME(gating_hypotheses, expert_hypotheses), targets_test))

    for e in range(epoch_count):
        gating_l2_states, gating_hypotheses, expert_l2_states, expert_hypotheses = feed_forward_ME(data_test)
        gl2_grad, gs_grad, el2_grad, el3_grad = gradient_ME(gating_l2_states, gating_hypotheses, expert_l2_states, expert_hypotheses, data_test, targets_test)
        gating_l2_weights -= learning_rate * gl2_grad
        gating_softmax_weights -= learning_rate * gs_grad
        expert_l2_weights -= learning_rate * el2_grad
        expert_l3_weights -= learning_rate * el3_grad

    _, gating_hypotheses, _, expert_hypotheses = feed_forward_ME(data_test)
    expert, counts = np.unique(gating_hypotheses.argmax(1), return_counts = True)
    print('Test set cost after training:', mean_squared_error_ME(gating_hypotheses, expert_hypotheses, targets_test))
    print('Test set classification error after training: %0.2f%%' % classification_error(hypothesis_ME(gating_hypotheses, expert_hypotheses), targets_test))
    print('% of examples where an expert was chosen as the lead:')
    for i in range(len(expert)):
        print('Expert', expert[i] + 1, '-', '%0.2f%%' % (100 * counts[i] / gating_hypotheses.shape[0]))

################################
# Train a mixture of experts
################################

gating_l2_weights = np.random.random((197, 25)) - 0.5
gating_softmax_weights = np.random.random((26, 10)) - 0.5

expert_l2_weights = np.random.random((10, 197, 25)) - 0.5
expert_l3_weights = np.random.random((10, 26, 10)) - 0.5

print('Training mixture of experts, might take about 20 minutes depending on your hardware...\n')
train_mixture_of_experts(3000, 10)
