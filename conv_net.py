import numpy as np
from sklearn.datasets import fetch_mldata

notice = '''
**************************************************************************
The purpose of this convolutional net, along with other neural networks
in this repository, has been first of all to facilitate learning and
only as a secondary objective to be fit for the task at hand.

I implemented this NN in order to learn the math behind convolutional
and pooling layers. The architecture of this NN, especially given the
fully connected layer between the max pooling layer and the softmax layer
(which further increases the number of parameters) is not great.

Nonetheless, this convolutional neural network, implemented solely using
basic matrix operations, works and can learn. The best result I achieved
on my laptop was a 6.98% classification error on the test set after training
for 2 epochs. The training took around 4 hours.

If you execute the script, the values set as default will run the network
for one full epoch. The running time depends on you hardware however it
might take as much as two hours or longer.
**************************************************************************
'''
print(notice)

################################
# Hyperparameters
################################

# Count of hidden units in the fully connected layer
no_of_hidden_units = 30

################################
# Preparing the data set
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
data_train = dataset.data[ids[:n_rows_train], :]
targets_train = np.zeros((n_rows_train, 10))
targets_train[np.arange(n_rows_train), dataset.target[ids[:n_rows_train]].astype(int)] = 1
data_test = dataset.data[ids[n_rows_train:], :]
targets_test = np.zeros((n_rows_test, 10))
targets_test[np.arange(n_rows_test), dataset.target[ids[n_rows_train:]].astype(int)] = 1

################################
# Weights initialization
################################

print('Initializing weights\n')
# Initializing a set of weights for each of the 6 feature maps
c1_weights = np.random.random((6, 5, 5)) - 0.5
# Initializing weights for the fully connected layer before softmax
f3_weights = np.random.random((865, no_of_hidden_units)) - 0.5
# Initializing weights connecting the fully connected layer to softmax group
softmax_weights = np.random.random((no_of_hidden_units + 1, 10)) - 0.5

def reset_weights():
    global c1_weights, f3_weights, softmax_weights
    # Initializing a set of weights for each of the 6 feature maps
    c1_weights = np.random.random((6, 5, 5)) - 0.5
    # Initializing weights for the fully connected layer before softmax
    f3_weights = np.random.random((865, no_of_hidden_units)) - 0.5
    # Initializing weights connecting the fully connected layer to softmax group
    softmax_weights = np.random.random((no_of_hidden_units + 1, 10)) - 0.5

################################
# Function definitions
################################

def forward_pass(image):
    # runs the network forward on a single example returning
    # layer states along the way including final output

    image = image.reshape((28,28))

    # numpy likely features a less trivial and likely faster way to perform
    # convolutions, but the point of this repo is to stay as close to the
    # metal as feasible, hence sticking to basic numpy operations
    c1 = np.zeros((6, 24, 24))
    for c in range(6):
        for i in range(24):
            for j in range(24):
                c1[c, i, j] = np.sum(image[i:i+5, j:j+5] * c1_weights[c, :, :])

    s2 = np.zeros((6, 12, 12))
    s2_idx = np.ndarray((6, 12, 12, 2))
    for s in range(6):
        for i in range(12):
            for j in range(12):
                max_ = np.max(c1[s, 2*i:2*i+2, 2*j:2*j+2])
                s2[s, i, j] = max_
                # There will be situations, mainly when all four entries are
                # equal to 0, where there will be multiple entries == max_.
                # In such a case, we will pick the first one returned.
                max_idx = np.where(c1[s, 2*i:2*i+2, 2*j:2*j+2] == max_)
                s2_idx[s, i, j, 0] = max_idx[0][0] + 2*i
                s2_idx[s, i, j, 1] = max_idx[1][0] + 2*j

    # Introducing the non-linearity at the max pooling layer
    s2 = sigmoid(s2)

    # Feeding forward through the fully connected layer
    z_f3 = np.c_[1, s2.reshape((1, -1))].dot(f3_weights)
    f3 = sigmoid(z_f3)

    # Connecting the fully connected layer to the softmax group
    z_softmax = np.c_[1, f3].dot(softmax_weights)
    # avoiding overflow
    z_softmax -= z_softmax.max()

    hypothesis = np.exp(z_softmax) / np.sum(np.exp(z_softmax))
    return c1, s2, s2_idx, f3, hypothesis

def gradient(image, target):
    c1, s2, s2_idx, f3, hypothesis = forward_pass(image)
    s2_unrolled = s2.reshape((-1, 1))
    image = image.reshape((28,28))

    delta_softmax = hypothesis - target
    softmax_weights_grad = np.outer(
        np.r_[np.atleast_2d(1), f3.T],
        delta_softmax
    )

    f3_grad = delta_softmax.dot(softmax_weights[1:, :].T)
    delta_f3 = f3_grad * f3 * (1 - f3)
    f3_weights_grad = np.outer(
            np.r_[np.atleast_2d(1), s2_unrolled],
            delta_f3
    )

    s2_grad = f3_weights[1:, :].dot(delta_f3.T)
    delta_s2 = s2_grad * s2_unrolled * (1 - s2_unrolled)
    delta_s2 = delta_s2.reshape((6, 12, 12))

    c1_weights_grad = np.zeros(c1_weights.shape)
    delta_c1 = np.zeros(c1.shape)
    for c in range(delta_s2.shape[0]):
        for i in range(delta_s2.shape[1]):
            for j in range(delta_s2.shape[2]):
                x, y = s2_idx[c, i, j].astype(int)
                delta_c1[c, x, y] = delta_s2[c, i, j]

    for c in range(c1.shape[0]):
        for i in range(c1.shape[1]):
            for j in range(c1.shape[2]):
                c1_weights_grad[c] += image[i:i+5, j:j+5] * delta_c1[c, i, j]

    return c1_weights_grad, f3_weights_grad, softmax_weights_grad

def numerical_gradient(image, target):
    delta = 1e-4
    c1_weights_grad = np.zeros(c1_weights.shape)
    for c in range(c1_weights.shape[0]):
        for i in range(c1_weights.shape[1]):
            for j in range(c1_weights.shape[2]):
                c1_weights[c][i][j] -= delta
                cost_minus = cost(forward_pass(image)[-1], target)

                c1_weights[c][i][j] += 2 * delta
                cost_plus = cost(forward_pass(image)[-1], target)

                c1_weights_grad[c][i][j] = (cost_plus - cost_minus) / (2 * delta)
                c1_weights[c][i][j] -= delta

    f3_weights_grad = np.zeros(f3_weights.shape)
    for i in range(f3_weights.shape[0]):
        for j in range(f3_weights.shape[1]):
            f3_weights[i][j] -= delta
            cost_minus = cost(forward_pass(image)[-1], target)

            f3_weights[i][j] += 2 * delta
            cost_plus = cost(forward_pass(image)[-1], target)

            f3_weights_grad[i][j] = (cost_plus - cost_minus) / (2 * delta)
            f3_weights[i][j] -= delta

    softmax_weights_grad = np.zeros(softmax_weights.shape)
    for i in range(softmax_weights.shape[0]):
        for j in range(softmax_weights.shape[1]):
            softmax_weights[i][j] -= delta
            cost_minus = cost(forward_pass(image)[-1], target)

            softmax_weights[i][j] += 2 * delta
            cost_plus = cost(forward_pass(image)[-1], target)

            softmax_weights_grad[i][j] = (cost_plus - cost_minus) / (2 * delta)
            softmax_weights[i][j] -= delta

    return c1_weights_grad, f3_weights_grad, softmax_weights_grad

def mean_cost(hypothesis_ary, targets):
    n = hypothesis_ary.shape[0]
    target_column_ids = targets.argmax(1)
    target_probabilities = hypothesis_ary[np.arange(n), target_column_ids]
    return - np.sum(np.log(target_probabilities)) / n

def classification_error(hypothesis_ary, targets):
    labels = targets.argmax(1)
    best_guesses = hypothesis_ary.argmax(1)
    return 100 - np.mean(labels == best_guesses).round(4) * 100

def cost_on_100_test_cases():
    h = hypotheses(data_test[:100, :])
    t = targets_test[:100, :]
    return mean_cost(h, t)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def hypotheses(data):
    h = [forward_pass(img)[-1] for img in data]
    return np.array(h).squeeze()

def train(epoch_count = 1, learning_rate = 0.01):
    global c1_weights, f3_weights, softmax_weights
    print('Training for', epoch_count, 'epoch(s) with learning_rate', learning_rate)
    for e in range(epoch_count):
        for i in range(data_train.shape[0]):
            c1_weights_grad, f3_weights_grad, softmax_weights_grad = gradient(data_train[i, :], targets_train[i, :])
            c1_weights -= learning_rate * c1_weights_grad
            f3_weights -= learning_rate * f3_weights_grad
            softmax_weights -= learning_rate * softmax_weights_grad
        print('Completed epoch', e + 1)
    print('Classification error on the test set: %0.2f%%' % classification_error(hypotheses(data_test), targets_test))

################################
# Training
################################

train(1, 0.01)
