import numpy as np

class RNN:
    def __init__(self, max_example_len = 5):
        # max_example_len is the maximum length of examples to generate
        self.max_example_len = max_example_len
        self.initial_state, self.wx, self.wRec = np.random.random(3) - 0.5

    def generate_example(self):
        # returns a tuple (lst, count)
        #   lst - list of length up to max_example_len containing 1s and 0s
        #   count - count of times 1 appears in the list
        length = np.random.randint(1, self.max_example_len + 1)
        lst = [1 if np.random.rand() > 0.5 else 0 for x in range(length)]
        return (lst, np.sum(lst))

    def forward_states(self, X):
        # returns a list consisting of states for each time step
        # states[-1] is the final activation that we output
        states = []
        previous_state = self.initial_state
        for x in X:
            states.append(self.update_state(x, previous_state))
            previous_state = states[-1]
        return states

    def update_state(self, x, previous_state):
        return previous_state * self.wRec + x * self.wx

    def train(self, epoch_count = 1, learning_rate = 0.01):
        for i in range(epoch_count):
            example = self.generate_example()
            partials = self.gradient(example)
            self.wRec -= learning_rate * partials[0]
            self.wx -= learning_rate * partials[1]
            self.initial_state -= learning_rate * partials[2]

    def cost(self, t, y):
        return (t - y) ** 2

    def run_example(self, example = None):
        # generates an example, runs the network forward and
        # prints out diagnostic information
        if not example:
            example = self.generate_example()
        X, target = example
        output = self.forward_states(X)[-1]
        print('Example:', X)
        print('Expected output:',target)
        print('Our output:', output, 'cost:', self.cost(target, output))
        print('---------------------')

    def gradient(self, example):
        X, target = example
        states = self.forward_states(X)
        deriv_cost_wrt_final_state = -2 * (target - states[-1])

        delta = deriv_cost_wrt_final_state
        partials_wx = []
        partials_wRec = []
        states_minus_1 = [self.initial_state] + states[:-1]
        for x, st in zip(reversed(X), reversed(states_minus_1)):
            partials_wx.append(delta * x)
            partials_wRec.append(delta * st)
            delta = delta * self.wRec
        partial_derviative_wrt_init_state = delta
        return sum(partials_wRec), sum(partials_wx), partial_derviative_wrt_init_state


    def numerical_gradient(self, example):
        # returns a tuple of partial derivatives of the cost function with
        # respect to wRec, wx and initial_state (we could 'cheat' and initiate
        # the initial state to 0, but learning it is more interesting)
        delta = 10e-4
        X, target = example

        self.wRec -= delta
        y = self.forward_states(X)[-1]
        c_minus = self.cost(target, y)
        self.wRec += 2*delta
        y = self.forward_states(X)[-1]
        c_plus = self.cost(target, y)
        partial_wRec = (c_plus - c_minus) / (2 * delta)
        self.wRec -= delta

        self.wx -= delta
        y = self.forward_states(X)[-1]
        c_minus = self.cost(target, y)
        self.wx += 2*delta
        y = self.forward_states(X)[-1]
        c_plus = self.cost(target, y)
        partial_wx = (c_plus - c_minus) / (2 * delta)
        self.wx -= delta

        self.initial_state -= delta
        y = self.forward_states(X)[-1]
        c_minus = self.cost(target, y)
        self.initial_state += 2*delta
        y = self.forward_states(X)[-1]
        c_plus = self.cost(target, y)
        partial_init_state = (c_plus - c_minus) / (2 * delta)
        self.initial_state -= delta

        return (partial_wRec, partial_wx, partial_init_state)


rnn = RNN()
print('Randomly initializing weights and the initial state:', \
'wRec:', rnn.wRec, 'wx:', rnn.wx, 'initial_state:', rnn.initial_state, '\n')
print('Generating 3 examples and running them before training.')
e1 = rnn.generate_example()
e2 = rnn.generate_example()
e3 = rnn.generate_example()
rnn.run_example(e1)
rnn.run_example(e2)
rnn.run_example(e3)

print('\n\nTraining on 1000 randomly generated examples.\n\n')
rnn.train(1000)
print('Weights and initial_state after training:', \
'wRec:', rnn.wRec, 'wx:', rnn.wx, 'initial_state:', rnn.initial_state, '\n')

print('Running the same examples we ran earlier, this time after training.')
e1 = rnn.generate_example()
e2 = rnn.generate_example()
e3 = rnn.generate_example()
rnn.run_example(e1)
rnn.run_example(e2)
rnn.run_example(e3)
