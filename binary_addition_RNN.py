'''
Implementing this RNN has been a very good learning experience and it is
interesting to experiment with the network. Nonetheless, the architectural
choices made during the implementation limit where one can take this RNN further.

In particular, the online approach to learning (updating weights after a single example)
and the way training examples are presented make the training process slow(*) which doesn't
lend itself to experimentation. Further to that, we are hitting a time barrier where
training becomes prohibitively expensive before hitting the disappearing / exploding
gradient problem.

Picking an online architecture also limits the choices available for improving learning.
It would be interesting to see to what extent I would be able to leverage RPROP to
help with the exploding / disappearing gradient problem and to what extent it could
speed up training, however given the online approach this is out of the question.
Implementing RMSPROP with the little experience that I have with it and given the
mini-batches sizes of 1 also doesn't seem to appealing.

Further to that, the way examples are generated precludes me from training the network
on shorter sequences and than running it on sequences of arbitrary length. I suspect
that there should be no surprises and with 3 hidden units such learning should
generalize fully to longer sequences, but this would not be easy to perform either.

In summary, there are multiple ways how through significant changes to the architecture
one could experiment with this RNN to a far greater extent. At this point, I think
I have gotten the bulk of the utility of implementing this RNN as is and feel that
my time will be better spent moving on to other material and hence chosing to leave
this RNN as is.

* training the RNN on 100 000 examples with example length of 5 takes nearly
3 minutes on my laptop
'''

import numpy as np

class RNN:
  # A recurrent neural network that learns binary addition. To simplify, will
  # perform addition left to right (least significant digit will be the first
  # in sequence)
    def __init__(self, example_len = 5, hidden_unit_count = 3, learning_rate = 1):
      self.learning_rate = learning_rate

      self.example_len = example_len
      self.example_max_value  = 2 * (2**(self.example_len - 1)) - 1
      self.examples = self.generate_example_dict()
      self.st_init = np.random.random(hidden_unit_count) - 0.5

      self.weights = {}
      self.weights['wx'] = np.random.random((3, hidden_unit_count)) - 0.5
      self.weights['wRec'] = np.random.random((hidden_unit_count + 1, hidden_unit_count)) - 0.5
      self.weights['wHidden'] = np.random.random((hidden_unit_count + 1, 1)) - 0.5

    def generate_example_dict(self):
      values = np.arange(self.example_max_value + 1)
      return {v: format(v, '0' + str(self.example_len + 1) + 'b')[::-1] for v in values}

    def generate_example(self):
        a, b = np.random.randint(self.example_max_value + 1, size=2)
        a_binary, b_binary = self.examples[a], self.examples[b]
        sum_ = a + b
        sum_binary = format(sum_, '0' + str(self.example_len + 1) + 'b')[::-1]
        X = [(int(a), int(b)) for a, b in zip(a_binary, b_binary)]
        target = np.array([int(t) for t in sum_binary])
        return X, target

    def fprop_in_time(self, X):
      Y = []
      hidden_layer_states = [self.st_init]
      for x1, x2 in X:
        st = self.sigmoid(np.array([1, x1, x2]).dot(self.weights['wx']) + np.r_[1, hidden_layer_states[-1]].dot(self.weights['wRec']))
        hidden_layer_states.append(st)
        y = self.sigmoid(np.r_[1, st].dot(self.weights['wHidden']))[0]
        Y.append(y)
      return hidden_layer_states, np.array(Y)

    def sigmoid(self, z):
      return 1 / (1 + np.exp(-z))

    def cost(self, Y, target):
      return  - np.sum(target * np.log(Y) + (1 - target) * np.log(1 - Y))

    def run_example(self, example = None):
      if not example:
        example = self.generate_example()
      X, target = example
      Y = self.fprop_in_time(X)[-1]
      cost = self.cost(Y, target)
      print('Number A:', [x[0] for x in X])
      print('Number B:', [x[1] for x in X])
      print('Expected output:', target)
      print('Our output:', np.round(Y, 2))
      print('Cost:', cost)
      print('-------------------')

    def gradient(self, example):
      X, target = example
      hidden_layer_states, Y = self.fprop_in_time(X)
      grad = {}

      deltas_output = Y - target
      deltas_hidden = []

      grad['wHidden'] = np.zeros(self.weights['wHidden'].shape)
      for hl, delta in zip(hidden_layer_states[1:], deltas_output):
        grad['wHidden'] += delta * np.r_[1, hl].reshape((-1,1))
        delta_hidden = (self.weights['wHidden'] * delta).T[0, 1:] * hl * (1 - hl)
        deltas_hidden.append(delta_hidden)

      grad['wx'] = np.zeros(self.weights['wx'].shape)
      for xs, delta in zip(X, deltas_hidden):
        input_layer = np.r_[1, xs[0], xs[1]]
        grad['wx'] += np.outer(input_layer, delta)

      grad['wRec'] = np.zeros(self.weights['wRec'].shape)
      for st_minus_one, delta in zip(hidden_layer_states[:-1], deltas_hidden):
        grad['wRec'] += np.outer(np.r_[1, st_minus_one], delta)

      grad_st_init = deltas_hidden[0].dot(self.weights['wRec'].T)[1:]

      return grad, grad_st_init

    def numerical_gradient(self, example):
      delta = 1e-4
      X, target = example

      grad = {}
      for k in self.weights:
        grad[k] = np.zeros(self.weights[k].shape)
        for i in range(self.weights[k].shape[0]):
          for j in range(self.weights[k].shape[1]):
            self.weights[k][i, j] -= delta
            cost_minus = self.cost(self.fprop_in_time(X)[-1], target)
            self.weights[k][i, j] += 2 * delta
            cost_plus = self.cost(self.fprop_in_time(X)[-1], target)
            self.weights[k][i, j] -= delta

            grad[k][i, j] = (cost_plus - cost_minus) / (2 * delta)

      return grad

    def update_weights(self, grad, grad_st_init):
      for k in self.weights:
        self.weights[k] -= self.learning_rate * grad[k]
      self.st_init -= self.learning_rate * grad_st_init

    def train(self, example_num = 1):
      for _ in range(example_num):
        example = self.generate_example()
        grad, grad_st_init = self.gradient(example)
        self.update_weights(grad, grad_st_init)



rnn = RNN(hidden_unit_count = 3, learning_rate = 0.2, example_len = 2)
notice = '\n!!!IMPORTANT!!! To simplify implementation this RNN performs \
binary addition from left to right!\n(examples are presented with least \
significant bit first)\n'
print(notice)
e1 = rnn.generate_example()
e2 = rnn.generate_example()
e3 = rnn.generate_example()
print('Performance on 3 randomly generated examples before training:')
rnn.run_example(e1)
rnn.run_example(e2)
rnn.run_example(e3)

print('\n\nTraining on 10000 examples...\n\n')
rnn.train(10000)

print('Performance on the same 3 examples after training:')
rnn.run_example(e1)
rnn.run_example(e2)
rnn.run_example(e3)
print('For additional information please see top of the Python script')
