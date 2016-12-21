from sklearn.datasets import fetch_mldata
import numpy as np

################################
# Set hyperparameters
################################

no_of_hidden_units = 200
learning_rate = 1
batch_size = 100

################################
# Prepare train and test sets
################################

# Fetching the dataset and performing minor normalization
# to help with training
print('Fetching MNIST dataset. Please wait...\n')
dataset = fetch_mldata('MNIST original', data_home='datasets')
dataset.data = dataset.data / 255

# Shuffling the ids to prepare for creation of train and test sets
ids = np.arange(len(dataset.data))
np.random.shuffle(ids)

# The full dataset consists of 70000 labelled examples.
# We will use 60000 examples for training and 10000 for our test set.
n_rows_train = 60000
n_rows_test = len(dataset.target) - n_rows_train
data_train = np.c_[np.ones((n_rows_train, 1)), dataset.data[ids[:n_rows_train], :]]
targets_train = np.zeros((n_rows_train, 10))
targets_train[np.arange(n_rows_train), dataset.target[ids[:n_rows_train]].astype(int)] = 1
data_test = np.c_[np.ones((n_rows_test, 1)), dataset.data[ids[n_rows_train:], :]]
targets_test = np.zeros((n_rows_test, 10))
targets_test[np.arange(n_rows_test), dataset.target[ids[n_rows_train:]].astype(int)] = 1

################################
# Initialize weights
################################

weights = {
    'input_to_hidden': np.random.rand(785, no_of_hidden_units) - 0.5,
    'hidden_to_softmax': np.random.rand(no_of_hidden_units + 1, 10) - 0.5
    }

def train(epoch_count = 1):
    print('Train set cost before training:', cross_entropy_cost(fprop(data_train)[-1], targets_train))
    for epoch in range(1, epoch_count + 1):
        batch_start_row = 0
        while batch_start_row < n_rows_train:
            grad = gradient(
                data_train[batch_start_row:batch_start_row + batch_size, :],
                targets_train[batch_start_row:batch_start_row + batch_size],
            )
            update_weights(grad, learning_rate)
            batch_start_row += batch_size
        print('Train set cost after epoch:', cross_entropy_cost(fprop(data_train)[-1], targets_train))
    print('--------------')
    print('Train set cost after training:', cross_entropy_cost(fprop(data_train)[-1], targets_train))
    print('Test set cost after training:', cross_entropy_cost(fprop(data_test)[-1], targets_test))
    print('train examples correctly classified: %0.2f%%' % (np.mean(fprop(data_train)[-1].argmax(1) == targets_train.argmax(1)) * 100))
    print('test examples correctly classified: %0.2f%%' % (np.mean(fprop(data_test)[-1].argmax(1) == targets_test.argmax(1)) * 100))

def fprop(batch):
    n_rows = len(batch)
    z2 = batch.dot(weights['input_to_hidden'])
    a2 = np.c_[np.ones((n_rows, 1)), sigmoid(z2)]
    z3 =  a2.dot(weights['hidden_to_softmax'])
    output = softmax_layer_activations(z3)
    return z2, a2, z3, output

def cross_entropy_cost(y, t):
    n_rows = len(t)
    return - np.sum(np.log(np.sum(t * y, 1))) / n_rows

def sigmoid(ary):
    return 1 / (1 + np.exp(-ary))

def softmax_layer_activations(batch):
    # avoiding overflow
    max_exponent = np.max(batch)
    batch -= max_exponent

    batch_exp = np.exp(batch)
    row_totals = np.sum(batch_exp, 1).reshape((-1, 1))
    return batch_exp / row_totals

def numerical_gradient(batch, targets):
    delta = 1e-4
    gradient = {}
    for k in weights:
        gradient[k] = np.empty(weights[k].shape)
        i, j = weights[k].shape
        for i in range(i):
            for j in range(j):
                weights[k][i, j] -= delta
                cost_minus = cross_entropy_cost(fprop(batch)[-1], targets)
                weights[k][i, j] += 2 * delta
                cost_plus = cross_entropy_cost(fprop(batch)[-1], targets)
                gradient[k][i, j] = (cost_plus - cost_minus) / (2 * delta)
                weights[k][i, j] -= delta
    return gradient

def gradient(batch, targets):
    m = len(targets)
    z2, a2, z3, output = fprop(batch)
    gradients = {}

    delta_a3 = output - targets
    gradients['hidden_to_softmax'] = a2.T.dot(delta_a3) / m

    delta_a2 = delta_a3.dot(weights['hidden_to_softmax'].T) * a2 * (1 - a2)
    # dropping the delta term for the bias unit
    delta_a2 = delta_a2[:, 1:]

    gradients['input_to_hidden'] = batch.T.dot(delta_a2) / m
    return gradients

def update_weights(gradient, learning_rate):
    for k in weights:
        weights[k] -= learning_rate * gradient[k]


print('Training for 15 epochs with learning_rate = 1')
learning_rate = 1
train(15)

print('\nTraining for 15 epochs with learning_rate = 0.1')
learning_rate = 0.1
train(5)
