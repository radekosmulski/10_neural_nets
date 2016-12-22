10 Neural Networks
------------------

An implementation of 10 simple neural networks in Matlab and Python. Each script can be executed - it will perform learning and will output diagnostic information. The implementations don't have to necessarily be particularly elegant nor efficient - the goal is to ensure I continue to practice.

1. XOR_network_with_one_hidden_unit.m

    A network with a hidden unit as described in [Parallel distributed models of schemata and sequential thought processes.](http://www.cs.toronto.edu/~hinton/absps/pdp8.pdf) by Rumelhart, D. E., Hinton, G. E., and Williams, R. J.  (figure 2).

    Learning Objectives:
    * get my feet wet with Matlab
    * practice applying the generalized delta rule

2. Parity_network_with_one_hidden_layer.m

    Another network outlined in [Parallel distributed models of schemata and sequential thought processes.](http://www.cs.toronto.edu/~hinton/absps/pdp8.pdf) by Rumelhart, D. E., Hinton, G. E., and Williams, R. J. the paper by Rumelhart, Hinton and Williams mentioned above. It outputs a 1 if the input pattern consisting of 4 binary digits contains an odd number of 1s and 0 otherwise.

    Learning Objectives:
    * practice implementing neural networks

3. Simple_softmax.m

    A very simple neural network for classification of input patterns into 3 categories with a softmax ouput layer.

    Learning Objectives:
    * learn to implement a softmax group

4. family_tree.py

    A neural network that learns distributed representations for family members and for relationships between them. Having learned to represent each person and each type of relationship through the microfeatures it infers, it is able to generalize to examples it has not seen before.

    The architecture of the neural network implemented here is described in [Learning distributed representations of concepts. Proceedings of the Eighth Annual Conference of the Cognitive Science Society, Amherst, Mass.](http://www.cs.toronto.edu/~hinton/absps/families.pdf) by Hinton, G. E. The architecture is not taken verbatim and some minor modifications have been made, such as switching to the cross entropy cost function. The dataset has been downloaded from [The UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Kinship).

    Learning Objectives:
    * learn about distributed representations
    * minimize a cost function with a complex error surface
    * understand why and how the acceleration / momentum method works
    * practice implementing a more complex neural network

5. count_ones_RNN.py

    A toy RNN for counting the number of times one appears in a list containing zeros and ones.

    I implemented this neural network based on a great tutorial by Peter Roelants that can be found [here](http://peterroelants.github.io/posts/rnn_implementation_part01/).

    Learning Objectives:
    * learn about and experiment with recurrent neural networks

6. digit_recognition.py

    A very simple take on digit recognition using the MNIST dataset. The network features a single hidden layer with variable number of hidden units feeding into a softmax output layer. The NN makes it very easy to experiment with parameters such as the learning_rate or batch_size. With 200 hidden units it consistently achieves a classification error of under 2.5% on the test set.

    Learning Ojbectives:
    * practice implementing a neural network with a non-trivial softmax group
    * experiment with the MNIST dataset in preperation of implementing a convolutional NN

7. binary_addition_RNN.py

    A slightly more complex RNN for performing binary addition. The network learns both the 3 set of weights and the initial state.

    Learning Objective:
    * practice implementing RNNs
