import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

################################
# Hyperparameters
################################

no_of_hidden_units = 1000
batch_size = 100

################################
# Preparing train and test sets
################################

# Fetching the dataset and performing minor normalization
# to help with training
print('\nFetching MNIST dataset. Please wait...\n')
dataset = fetch_mldata('MNIST original', data_home='datasets')
dataset.data = dataset.data / 255

# Shuffling the ids to prepare for creation of train and test sets
ids = np.arange(len(dataset.data))
np.random.shuffle(ids)

# The full dataset consists of 70000 labelled examples.
# We will use 60000 examples for training and 10000 for our test set.
n_rows_train = 60000
n_rows_test = len(dataset.target) - n_rows_train
targets_train = np.zeros((n_rows_train, 10))
targets_train[np.arange(n_rows_train), dataset.target[ids[:n_rows_train]].astype(int)] = 1
joint_train = np.c_[targets_train, dataset.data[ids[:n_rows_train], :]]
del targets_train

data_test = dataset.data[ids[n_rows_train:], :]
targets_test = np.zeros((n_rows_test, 10))
targets_test[np.arange(n_rows_test), dataset.target[ids[n_rows_train:]].astype(int)] = 1
joint_test = np.c_[targets_test, data_test]

################################
# Initializing weights
################################

print('Initializing weights')
weights = np.random.normal(0, 0.01, (794, no_of_hidden_units))
delta_weights = np.zeros(weights.shape)
# I will not be using biases. This is my first fully independent implementation
# of an RBM - first need to get the basics down.

################################
# Function definitions
################################

def visible_states_to_hidden_probabilities(visible_states):
    return sigmoid(visible_states.dot(weights))

def hidden_states_to_visible_probabilities(hidden_states):
    return sigmoid(hidden_states.dot(weights.T))

def probabilities_to_binary_states(probabilities):
    return np.random.binomial(1, probabilities)

def sigmoid(xs):
    return 1/(1 + np.exp(-xs))

def test_set_average_free_energy():
    xs = joint_test.dot(weights)
    return - np.sum(np.log(1 + np.exp(xs))).round(2) / n_rows_test

def plot_20_receptive_fields():
    randomly_chosen_receptive_fields = np.random.choice(np.arange(no_of_hidden_units), size=20, replace=False)
    for i, r in enumerate(randomly_chosen_receptive_fields):
        plt.subplot(4, 5, i+1)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(weights[10:, r].reshape((28,28)), cmap='gray')
    plt.show()

def train(epoch_count = 1, momentum = 0.9, acceleration = 0.6):
    global weights, fantasy_hidden_states, fantasy_visible_states, fantasy_hidden_probabilities, delta_weights
    print('Training for', epoch_count, 'epoch(s) with momentum', momentum, 'and acceleration', acceleration)
    for i in range(epoch_count):
        batch_start_row = 0
        while batch_start_row < n_rows_train:

            # collecting positive statistics
            visible_probabilities = joint_train[batch_start_row:batch_start_row + batch_size, :]
            visible_states = probabilities_to_binary_states(visible_probabilities)
            hidden_states_probabilities = visible_states_to_hidden_probabilities(visible_states)

            # updating fantasy particles
            fantasy_visible_probabilities = hidden_states_to_visible_probabilities(fantasy_hidden_states)
            fantasy_visible_states = probabilities_to_binary_states(fantasy_visible_probabilities)
            fantasy_hidden_probabilities = visible_states_to_hidden_probabilities(fantasy_visible_states)
            fantasy_hidden_states = probabilities_to_binary_states(fantasy_hidden_probabilities)

            # calculating gradient
            positive_gradient = visible_states.T.dot(hidden_states_probabilities) / batch_size
            negative_gradient = fantasy_visible_states.T.dot(fantasy_hidden_probabilities) / batch_size

            # updating weights
            delta_weights = momentum * delta_weights + acceleration * (positive_gradient - negative_gradient)
            weights += delta_weights
            # weights += learning_rate * (positive_gradient - negative_gradient)

            batch_start_row += batch_size
        print('Completed epoch', i + 1)

def test_set_classification_error():
    energies = []
    for i in range(10):
        target_vectors = np.zeros((n_rows_test, 10))
        target_vectors[:, i] = 1
        xs = np.c_[target_vectors, data_test].dot(weights)
        energies.append(-np.sum(np.log(1 + np.exp(xs)), 1))
    energies = np.array(energies).T
    return 1 - np.average(energies.argmin(1) == targets_test.argmax(1))

################################
# Initializing fantasy particles
################################

print('Initializing fantasy particles\n')
fantasy_visible_states = probabilities_to_binary_states(joint_train[:batch_size, :])
for i in range(100):
    fantasy_hidden_probabilities = visible_states_to_hidden_probabilities(fantasy_visible_states)
    fantasy_hidden_states = probabilities_to_binary_states(fantasy_hidden_probabilities)
    fantasy_visible_probabilities = hidden_states_to_visible_probabilities(fantasy_hidden_states)
    fantasy_visible_states = probabilities_to_binary_states(fantasy_visible_probabilities)

################################
# Performing training
################################

print('Hidden unit count is set to', no_of_hidden_units, 'and batch size is set to', batch_size, '\n')
train(7, 0.4, 0.05)
print('Test set classification error: %0.2f%%' % (test_set_classification_error().round(4) * 100))
print('Plotting 20 randomly selected receptive fields...')
plot_20_receptive_fields()
