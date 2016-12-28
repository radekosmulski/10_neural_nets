10 Neural Networks
------------------

An implementation of 10 simple neural networks in Matlab and Python. Each script can be executed - it will perform learning and will output diagnostic information. The implementations don't have to necessarily be particularly elegant nor efficient - the goal is to ensure I continue to practice.

I worked on these neural networks while part taking in a [Neural Networks for Machine Learning MOOC](https://www.coursera.org/learn/neural-networks) taught by Professor Geoffrey Hinton. This is an outstanding course. The implementations are based on teachings by Professor Hinton and further readings that I reference.

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

    Learning Objectives:
    * practice implementing RNNs

8. mixture_of_experts.py

  A mixture of experts trained on the MNIST dataset. Each expert is a simple fully connected NN with one hidden layer consisting of 25 hidden units and an output layer consisting of 10 neurons. A gating network is trained to discern what mixture of experts should be utilized to produce output on a given example. Experts receive training signal in proportion to the belief of the gating network of their applicability to a given example (this promotes specialization).

    I implemented this mixture of experts based on [Adaptive Mixture of Local Experts](http://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf) by Jacobs, Jordan, Nowlan and Hinton.

    Learning Objectives:
    * practice implementing a model combining multiple NNs
    * understand the mathematics behind the gating network and how error derivatives are propagated to the experts

9. conv_net.py

    A small convolutional neural network using the MNIST dataset. It features a convolutional layer and a max-pooling layer feeding directly into a softmax. There have been numerous sources I used to implement this CNN, including:
    * [Lecun et al. 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
    * notes for the Stanford CS class [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/)
    * [A Beginner's Guide To Understanding Convolutional Neural Networks](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)

    Learning Objectives:
    * figure out how the error is backpropagated in a NN featuring convolutional and pooling layers

10. boltzmann.py

    A joint Restricted Boltzmann Machine (RBM) trained on the MNIST dataset using Persistent Contrastive Divergence with momentum. After training, in order to predict class membership of an example, the example is presented to the RBM along with each possible label. The free energy of each example - label pair is evaluated and the label of a combination with lowest free energy is picked as the most likely class.

    I implemented this RBM based on information contained in [Training Restricted Boltzmann Machines using Approximations to the Likelihood Gradient](http://www.cs.toronto.edu/~tijmen/pcd/pcd.pdf) by Tijmen Tieleman and [A practical guide to training restricted Boltzmann machines](http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf) by Geoffrey Hinton.

    Learning Objectives:
    * practice implementing a Restricted Boltzmann Machine
    * gain hands on experience with the Persistent Contrastive Divergence algorithm
