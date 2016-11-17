import numpy as np
import re

###############################################################################
# Set up
###############################################################################

# Reading in data
relationship = []
person_A = []
person_B = []

f = open('datasets/kinship.data')
for line in f.read().splitlines():
    if not line.rstrip(): continue
    matches = re.match(r"(\w+)\((\w+), (\w+)\)", line).groups()
    relationship.append(matches[0])
    person_A.append(matches[2])
    person_B.append(matches[1])


# Representing the dataset as a dict. This allows us to accomodate
# representing cases where a single person has 1 or more relatives
# of a given kind (2 is the maximimu for this datset).

dataset_as_dict = {}
for i in range(len(person_A)):
    key = (person_A[i], relationship[i])
    val = dataset_as_dict.get(key, [])
    val.append(person_B[i])
    dataset_as_dict[key] = val

# Mapping each person / relationship to a corresponding index.
person_mapping = {person: idx for idx, person in enumerate(set(person_A))}
relationship_mapping = \
        {rel: idx + 24 for idx, rel in enumerate(set(relationship))}

m = len(dataset_as_dict)

# Encoding the dataset into input / target pairs as specified in the paper.
input_vectors = np.zeros((m, 36))
target_vectors = np.zeros((m, 24))

for i, tup in enumerate(dataset_as_dict.items()):
    input_ = tup[0]
    target = tup[1]
    input_vectors[i, person_mapping[input_[0]]] = 1
    input_vectors[i, relationship_mapping[input_[1]]] = 1
    for relative in target:
        target_vectors[i, person_mapping[relative]] = 1

# Initializing the weights
weights = {
    'person_to_embed': np.random.rand(24, 6) * 0.66 - 0.33,
    'relationship_to_embed': np.random.rand(12, 6) * 0.66 - 0.33,
    'embed_to_hidden': np.random.rand(13, 12) * 0.66 - 0.33,
    'hidden_to_embed': np.random.rand(13, 6) * 0.66 - 0.33,
    'embed_to_output': np.random.rand(7, 24) * 0.66 - 0.33,
}

# Change in weights for initial epoch (since we do not have values for t - 1)
previous_delta_w = {k: np.zeros(weights[k].shape) for k in weights}

# Splitting the data into a train and test set
test_set_ids = np.random.randint(m, size=4)
test_set = input_vectors[test_set_ids, :], target_vectors[test_set_ids, :]
input_vectors = np.delete(input_vectors, test_set_ids, 0)
target_vectors = np.delete(target_vectors, test_set_ids, 0)

# Recalculating the m after the split
m = len(input_vectors)
###############################################################################
# Function definitions
###############################################################################

def forward_prop():
    embed_person = input_vectors[:, :24].dot(weights['person_to_embed'])
    embed_relationship = \
        input_vectors[:, 24:].dot(weights['relationship_to_embed'])
    embedding_layer_state = \
        np.c_[np.ones((m, 1)), embed_person, embed_relationship]

    inputs_to_hidden_layer = \
        embedding_layer_state.dot(weights['embed_to_hidden'])
    hidden_layer_state = np.c_[np.ones((m, 1)), sigmoid(inputs_to_hidden_layer)]

    inputs_to_output_embedding_layer = \
        hidden_layer_state.dot(weights['hidden_to_embed'])
    output_embedding_layer_state = \
        np.c_[np.ones((m, 1)), inputs_to_output_embedding_layer]

    inputs_to_output_layer = \
        output_embedding_layer_state.dot(weights['embed_to_output'])
    output_layer_state = sigmoid(inputs_to_output_layer)
    return output_layer_state, output_embedding_layer_state, \
        hidden_layer_state, embedding_layer_state

def back_prop(output_layer_state, output_embedding_layer_state,
        hidden_layer_state, embedding_layer_state):

    gradient = {}

    delta_output_layer = output_layer_state - target_vectors
    gradient['embed_to_output'] = \
        output_embedding_layer_state.T.dot(delta_output_layer) / m

    delta_output_embedding_layer = \
        delta_output_layer.dot(weights['embed_to_output'].T)[:, 1:]
    gradient['hidden_to_embed'] = \
        hidden_layer_state.T.dot(delta_output_embedding_layer) / m

    deriv_hidden_layer = \
        delta_output_embedding_layer.dot(weights['hidden_to_embed'].T)[:, 1:]
    delta_hidden_layer =\
        deriv_hidden_layer * hidden_layer_state[:, 1:] * (1 - hidden_layer_state[:, 1:])
    gradient['embed_to_hidden'] = \
        embedding_layer_state.T.dot(delta_hidden_layer) / m

    delta_embedding_layer = \
        delta_hidden_layer.dot(weights['embed_to_hidden'].T)[:, 1:]
    gradient['relationship_to_embed'] = \
        input_vectors[:, 24:].T.dot(delta_embedding_layer[:, 6:]) / m
    gradient['person_to_embed'] = \
        input_vectors[:, :24].T.dot(delta_embedding_layer[:, :6]) / m
    return gradient

def cost(output_layer_state):
    return -np.sum(target_vectors * np.log(output_layer_state) + \
            (1 - target_vectors) * np.log(1 - output_layer_state)) / m

def sigmoid(ary):
    return 1 / (1 + np.exp(-ary))

def numerical_gradient():
    delta = 1e-4
    gradient = {}
    for k in weights:
        gradient[k] = np.empty(weights[k].shape)
        i, j = weights[k].shape
        for i in range(i):
            for j in range(j):
                weights[k][i, j] -= delta
                c1 = cost(forward_prop()[0])
                weights[k][i, j] += 2 * delta
                c2 = cost(forward_prop()[0])
                gradient[k][i, j] = (c2 - c1) / (2 * delta)
                weights[k][i, j] -= delta
    return gradient

def delta_weights(gradient, previous_delta_w, acceleration, momentum):
    delta_w = {}
    for k in gradient:
        delta_w[k] = \
            - acceleration * gradient[k] + momentum * previous_delta_w[k]
    return delta_w

def update_weights(delta_w):
    global weights
    for k in weights:
        weights[k] +=  delta_w[k]
    return weights

def train(acceleration, momentum, epoch_count):
    global previous_delta_w

    for i in range(epoch_count):
        layer_states = forward_prop()
        gradient = back_prop(*layer_states)
        current_delta_weights = delta_weights(gradient, previous_delta_w, acceleration, momentum)
        update_weights(current_delta_weights)
        previous_delta_w = current_delta_weights
    pass

def reset_weights():
    # useful function for experimentation and picking hyperparameters
    global weights
    weights = {
        'person_to_embed': np.random.rand(24, 6) * 0.66 - 0.33,
        'relationship_to_embed': np.random.rand(12, 6) * 0.66 - 0.33,
        'embed_to_hidden': np.random.rand(13, 12) * 0.66 - 0.33,
        'hidden_to_embed': np.random.rand(13, 6) * 0.66 - 0.33,
        'embed_to_output': np.random.rand(7, 24) * 0.66 - 0.33,
    }

def incorrectly_identified_relatives_count():
    guessed_relatives = np.where(forward_prop()[0] > 0.5, 1, 0)
    error_count = 0
    for i in range(m):
        if not np.all(guessed_relatives[i, :] == target_vectors[i, :]): error_count += 1
    return error_count
###############################################################################
# Calculations
###############################################################################

print('Cost before training:', cost(forward_prop()[0]))
train(0.6, 0.05, 20)
train(0.2, 0.95, 600)
print('Cost after training:', cost(forward_prop()[0]))
print('Errors on train set:', incorrectly_identified_relatives_count(), 'out of', m)
# This is a bit hackish... but at this point not sure there is value
# in making significant changes to the code above just to achieve this...
input_vectors = test_set[0]
target_vectors = test_set[1]
m = 4
print('Errors on test set:', incorrectly_identified_relatives_count(), 'out of', m)
