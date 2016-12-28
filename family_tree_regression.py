import numpy as np
import re
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

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

# Splitting the data into a train and test set
test_set_ids = np.random.randint(m, size=4)
test_input = input_vectors[test_set_ids, :]
test_target = target_vectors[test_set_ids, :]
input_vectors = np.delete(input_vectors, test_set_ids, 0)
target_vectors = np.delete(target_vectors, test_set_ids, 0)

# Recalculating the m after the split
m = len(input_vectors)
###############################################################################
# Function definitions
###############################################################################

# build and fit model

model = LogisticRegression()
poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(input_vectors)
y = np.argmax(target_vectors, axis=1)
model.fit(X, y)
y_hat = model.predict(X)

print('in-sample error rate = {}'.format(np.mean(np.abs(y_hat-y) > 0)))

X_test = poly.fit_transform(test_input)
y_test = np.argmax(test_input, axis=1)
y_test_hat = model.predict(X_test)
print('out-of-sample error rate = {}'.format(np.mean(np.abs(y_test_hat-y_test) > 0)))
print('target y:', y_test)
print('predicted y:', y_test_hat)

