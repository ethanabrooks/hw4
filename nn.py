"""
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
"""

from numpy import roll, squeeze, array
from numpy import random, zeros, ones, matrix, unique, argmax, fromfunction, vectorize, where, concatenate, \
    repeat, vstack, dot
from numpy.core.umath import square, multiply, log
from numpy.testing import assert_almost_equal
from scipy.special._ufuncs import expit
from sklearn.metrics import accuracy_score


def get_width(m):
    if len(m.shape) == 1:
        return m.size
    return m.shape[1]


def feed_forward_once(inputs, theta):
    """
    :param inputs: matrix [[ n x d+1 ]]
    :param theta: matrix [[ d+1 x d ]]
    :return matrix [[ n x d ]]
    """
    d1, d = theta.shape
    assert get_width(inputs) == d1
    dot_product = dot(inputs, theta)
    return expit(dot_product)


def init_thetas(epsilon, num_hl, num_features, hl_width, num_classes, rand=True):
    def generate(size):
        if rand:
            return random.uniform(-epsilon, epsilon, size=size)
        else:
            return ones(size)

    first = [[num_features + 1, hl_width]]
    last = [[hl_width + 1, num_classes]]
    if num_hl == 0:
        shapes = matrix([num_features + 1, num_classes])
    elif num_hl == 1:
        shapes = vstack([first, last])
    else:
        shapes = vstack([
            first,
            repeat([[hl_width + 1, hl_width]], num_hl, axis=0),
            last
    ])
    size = dot(shapes[:, 0], shapes[:, 1])
    thetas = generate(size)
    return thetas, shapes


def feed_forward(input, thetas, thetas_shapes):
    a_shapes = thetas_shapes[:, 0].ravel()
    activations = ones(sum(a_shapes))
    a_list = split_into_list(activations, a_shapes)
    for l, theta in enumerate(split_into_list(thetas, thetas_shapes)):
        a_list[l][1:] = input
        assert a_list[l][0] == 1
        input = feed_forward_once(a_list[l], theta)
    return concatenate(a_list), a_shapes, input  # = output


def feed_forward_multiple_inputs(inputs, thetas, thetas_shapes):
    """
    :param inputs: matrix [[ n x d ]]
    :param thetas: matrix [[ num_layers x d(d+1) ]]
    :return: matrix [[ n x d ]]
    assumes num classes = d !
    """
    a_shapes = roll(thetas_shapes, 1, axis=1)
    a_shapes[:, 0] = inputs.shape[0]
    activations = ones(dot(a_shapes[:, 0], a_shapes[:, 1]))
    a_list = split_into_list(activations, a_shapes)
    for l, theta in enumerate(split_into_list(thetas, thetas_shapes)):
        a_list[l][:, 1:] = inputs
        inputs = feed_forward_once(a_list[l], theta)
    return inputs  # = outputs


def get_error(output, classes, y_i):
    error = output.copy()
    error[where(classes == y_i)] = output[where(classes == y_i)] - 1
    return error


def split_into_list(vector, shapes):
    vector = squeeze(array(vector))
    list = []
    for shape in shapes:
        size = shape.prod()
        shape = squeeze(array(shape))
        list.append(vector[:size].reshape(shape))
        vector = roll(vector, size)
    return list


# def recover(theta, d=None):
#     d1 = floor(sqrt(theta.size)) + 1
#     theta_ = theta.reshape(d1, -1)
#     return theta_ if d is None else theta_[:, :d]


class NeuralNet:
    def __init__(self, layers, epsilon=0.12, learningRate=.001, numEpochs=100, gradientChecking=False, randTheta=True):
        """
        Constructor
        Arguments:
        	layers - a numpy array of L-2 integers (L is # layers in the network)
        	epsilon - one half the interval around zero for setting the initial weights
        	learningRate - the learning rate for backpropagation
        	numEpochs - the number of epochs to run during training
        """
        self.num_hl = layers  # layers for theta is hidden layers + 1
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.numEpochs = numEpochs
        self.gradientChecking = gradientChecking
        self.randTheta = randTheta
        self.hl_width = 3
        self.reg_factor = .0001

    def get_gradients(self, X, y):
        gradients = zeros(self.thetas.shape)
        thetas_list = split_into_list(self.thetas, self.thetas_shapes)
        for i, instance in enumerate(X):
            activations, a_shapes, output = feed_forward(
                instance, self.thetas, self.thetas_shapes
            )
            g_primes = get_g_prime(activations, a_shapes)
            error = get_error(output, self.classes, y[i])
            deltas = get_deltas(g_primes, thetas_list, error)
            a_list = split_into_list(activations, a_shapes)
            update_gradient(gradients, a_list, deltas)
        return gradients / X.shape[0]

    def multicost(self, X, y_bin, c):
        """
        :param X: inputs
        :param y_bin: [[ n x K ]] of bin values
        :param perturbed_thetas: [[ theta.size x theta.size ]]
            each row of perturbed_thetas is equal to theta.ravel()
            with one perturbed value
        :return: [[ theta.shape ]] row i corresponding to
        the cost after perturbing the ith value in theta
        """

        def cost(i, j):
            perturbed_thetas = thetas.copy()
            perturbed_thetas[i, j] += c
            predictions = feed_forward_multiple_inputs(X,
                                                       perturbed_thetas,
                                                       thetas_shapes)
            if j % 1000 == 0:
                print ".",
            return get_cost(
                X, y_bin, predictions[:, :self.classes.size], reg_factor, perturbed_thetas
            )

        thetas = self.thetas
        thetas_shapes = self.thetas_shapes
        reg_factor = self.reg_factor
        return fromfunction(vectorize(cost), self.thetas.shape)

    def check_gradient(self, gradient, X, y, c=.0001):
        y_bin = get_y_bin(y, self.classes)
        cost_plus, cost_minus = (self.multicost(
            X, y_bin, self.thetas, c_,
        ) for c_ in (c, -c))
        grad_approx = get_grad_approx(c, cost_minus, cost_plus)
        assert_almost_equal(gradient, grad_approx, decimal=4)
        print "Gradient checks out."

    def fit(self, X, y):
        """
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        NOTE: num_features must exceed num_classes
        """
        self.classes = unique(y)
        self.thetas, self.thetas_shapes = init_thetas(
            self.epsilon,
            self.num_hl,
            X.shape[1],
            self.hl_width,
            self.classes.size,
            rand=self.randTheta
        )
        for _ in range(self.numEpochs):
            gradient = self.get_gradients(X, y)
            regularize_gradient(gradient, self.reg_factor, self.thetas)
            if self.gradientChecking:
                self.check_gradient(gradient, X, y)
            update_thetas(self.thetas, self.learningRate, gradient)
            print ".",

    def predict(self, X):
        """
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        """
        return feed_forward_multiple_inputs(X, self.thetas, self.classes.size)

    def score(self, X_train, y_train, X_test, y_test):
        self.fit(X_train, y_train)
        probabilities = self.predict(X_test)
        print
        y_pred = self.classes[argmax(probabilities, axis=1)]
        print y_pred
        print y_test
        return accuracy_score(y_test, y_pred)

    def visualizeHiddenNodes(self, filename):
        """
        CIS 519 ONLY - outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        """


def get_g_prime(activations, a_shapes):
    g_prime_with_bias = multiply(activations, 1 - activations)
    return [g_prime[1:]  # eliminate bias node
            for g_prime in split_into_list(g_prime_with_bias, a_shapes)]


def next_delta(thetas_list, deltas, g_prime, l):
    #  recover weights for lth layer, omitting bias nodes
    theta_no_bias = thetas_list[l][1:, :]
    return multiply(dot(deltas[-1], theta_no_bias.T), g_prime[l])


def get_deltas(g_primes, thetas_list, last_delta):
    """
    :param g_prime: sigma'(z), matrix [[ layers x d+1 ]]
    :param last_delta: y - a^L, array [ d+1 ]
    :return: deltas, matrix [[ layers x d+1 ]]
    """
    deltas = [last_delta]
    hidden_layer_indices = range(len(thetas_list) - 1, -1, -1)
    for l in hidden_layer_indices:
        deltas.append(next_delta(thetas_list, deltas, g_primes, l))
    deltas.reverse()
    return deltas


def get_gradient_update_val(activations, deltas, l):
    # rows ~ ouputs, columns ~ inputs
    activation_T = matrix(activations)[l, :].T  # TODO
    delta = deltas[l + 1, :]
    dot_product = activation_T * delta
    return dot_product.ravel()


def gradient_update_vector(a_list, deltas):
    num_layers, d1 = a_list.shape
    update_vals = []
    for l in range(num_layers):
        update_vals.append(get_gradient_update_val(a_list, deltas, l))
    return update_vals


def update_thetas(thetas, learningRate, gradient):
    thetas -= learningRate * gradient


def update_gradient(gradient, a_list, deltas):
    update_matrix = gradient_update_vector(a_list, deltas)
    gradient += update_matrix


def regularize_gradient(gradient, reg_factor, thetas):
    gradient[:, 1:] += reg_factor * thetas[:, 1:]


def get_y_bin(y, classes):
    y_bin = zeros((y.size, classes.size))
    Y = y.reshape(1, y.size)
    y_bin[classes == Y.T] = 1
    return y_bin


def calculate_cost_no_reg(y_bin, predictions):
    n = y_bin.shape[0]
    term1 = multiply(y_bin, log(predictions))
    term2 = multiply((1 - y_bin), log(1 - predictions))
    return -(term1.sum() + term2.sum()) / n


def calculate_cost_reg(reg_factor, thetas, n):
    return reg_factor / (2. * n) * square(thetas).sum()


def get_grad_approx(c, cost_minus, cost_plus):
    return (cost_plus - cost_minus) / (2. * c)


def get_cost(X, y_bin, predictions, reg_factor, thetas):
    no_reg = calculate_cost_no_reg(y_bin, predictions)
    reg = calculate_cost_reg(reg_factor, thetas, X.shape[0])
    return no_reg + reg
