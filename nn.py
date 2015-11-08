"""
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
"""
from numpy import random, zeros, ones, eye, matrix, apply_along_axis, unique, dot, argmax
from numpy.core.umath import square
from numpy.ma import exp, true_divide, multiply, log, masked_array, floor, sqrt, divide
from numpy.testing import assert_almost_equal
from sklearn.metrics import accuracy_score


def sigmoid(x):
    return true_divide(1, (1 + exp(-x)))


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

    return sigmoid(dot(inputs, theta))


def init_thetas(epsilon, layers, d, num_classes, rand=True):
    size = layers, (d+1) * d
    if rand:
        thetas_unmasked = random.uniform(-epsilon, epsilon, size=size)
    else:
        thetas_unmasked = ones(size)
    mask = zeros(thetas_unmasked.shape)
    start_mask = (d+1) * num_classes
    mask[-1, start_mask:] = 1
    return masked_array(data=thetas_unmasked, mask=mask, fill_value=0)


def feed_forward(input, thetas):
    activations = ones([thetas.shape[0], get_width(input) + 1])
    # L (including output layer), d+1 (for bias nodes)
    for l, theta in enumerate(thetas):
        activations[l, 1:] = input
        assert activations[l, 0] == 1
        input = feed_forward_once(activations[l, :], reshape(theta))
    return activations, input  # = output


def feed_forward_multiple_inputs(inputs, thetas):
    """
    :param inputs: matrix [[ n x d ]]
    :param thetas: matrix [[ num_layers x d(d+1) ]]
    :return: matrix [[ n x d ]]
    assumes num classes = d !
    """
    n, d = inputs.shape
    activations = ones([n, d + 1])  # d+1 for bias nodes
    for i, theta in enumerate(thetas):
        assert get_width(theta) == d * (d + 1)
        activations[:, 1:] = inputs
        inputs = feed_forward_once(activations, reshape(theta))
    return inputs  # = outputs


def get_error(output, classes, y_i):
    error = output.copy()
    error[classes == y_i] = output[classes == y_i] - 1
    return error


def reshape(theta):
    num_unmasked = masked_array(theta).count()
    d1 = floor(sqrt(theta.size)) + 1
    return theta[:num_unmasked].reshape(d1, num_unmasked / d1)


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
        self.layers = layers + 1  # layers for theta is hidden layers + 1
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.numEpochs = numEpochs
        self.gradientChecking = gradientChecking
        self.randTheta = randTheta
        self.reg_factor = 0  # TODO modify

    def get_gradients(self, X, y):
        gradients = self.thetas.copy()
        gradients.fill(0)
        for i, instance in enumerate(X):
            activations, output = feed_forward(instance, self.thetas)
            g_prime = get_g_prime(activations)
            error = get_error(output, self.classes, y[i])
            deltas = get_deltas(g_prime, self.thetas, error)
            update_gradient(gradients, activations, deltas)
        return gradients / X.shape[0]

    def check_gradient(self, gradient, X, y, c=.0001):
        y_bin = get_y_bin(y, self.classes)
        cost_plus, cost_minus = (self.multicost(
            X, y_bin, perturb(self.thetas, c_),
        ) for c_ in (c, -c))
        grad_approx = get_grad_approx(c, cost_minus, cost_plus)
        assert_almost_equal(gradient, grad_approx, decimal=4)

    def multicost(self, X, y_bin, perturbed_thetas):
        """
        :param X: inputs
        :param y_bin: [[ n x K ]] of bin values
        :param perturbed_thetas: [[ theta.size x theta.size ]]
            each row of perturbed_thetas is equal to theta.ravel()
            with one perturbed value
        :return: [[ theta.shape ]] row i corresponding to
        the cost after perturbing the ith value in theta
        """
        def cost(ravel_thetas, shape, reg_factor):
            thetas = ravel_thetas.reshape(shape)
            predictions = feed_forward_multiple_inputs(X, thetas)
            return get_cost(X, y_bin, predictions, reg_factor, thetas)

        cost_ravel = apply_along_axis(cost, 1,
                                      perturbed_thetas,
                                      self.thetas.shape,
                                      self.reg_factor)
        return cost_ravel.reshape(self.thetas.shape)

    def fit(self, X, y):
        """
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        """
        self.classes = unique(y)
        self.thetas = init_thetas(self.epsilon,
                                  self.layers,
                                  X.shape[1],
                                  self.classes.size,
                                  rand=self.randTheta)
        for _ in range(self.numEpochs):
            gradient = self.get_gradients(X, y)
            regularize_gradient(gradient, self.reg_factor, self.thetas)
            if self.gradientChecking:
                self.check_gradient(gradient, X, y)
            update_thetas(self.thetas, self.learningRate, gradient)

    def predict(self, X):
        """
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        """
        return feed_forward_multiple_inputs(X, self.thetas)

    def score(self, X_train, y_train, X_test, y_test):
        self.fit(X_train, y_train)
        probabilities = self.predict(X_test)
        y_pred = self.classes[argmax(probabilities)]
        return accuracy_score(y_test, y_pred)

    def visualizeHiddenNodes(self, filename):
        """
        CIS 519 ONLY - outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        """


def get_g_prime(activations):
    return multiply(activations, 1 - activations)[:, 1:]  # omit bias activations


def next_delta(thetas, deltas, g_prime, l):
    #  recover weights for lth layer, omitting bias nodes
    theta_no_bias = reshape(thetas[l, :])[1:, :]
    return multiply(dot(deltas[l + 1, :], theta_no_bias.T), g_prime[l, :])


def get_deltas(g_prime, thetas, last_delta):
    """
    :param g_prime: sigma'(z), matrix [[ layers x d+1 ]]
    :param thetas: matrix [[ layers x (d x d+1) ]]
    :param last_delta: y - a^L, array [ d+1 ]
    :return: deltas, matrix [[ layers x d+1 ]]
    """
    num_layers, d = g_prime.shape
    deltas = zeros([num_layers + 1, d])
    deltas[-1, :last_delta.size] = last_delta
    hidden_layer_indices = range(num_layers - 1, 0, -1)
    for l in hidden_layer_indices:
        deltas[l, :] = next_delta(thetas, deltas, g_prime, l)
    return deltas


def get_gradient_update_val(activations, deltas, l):
    # rows ~ ouputs, columns ~ inputs
    activation_T = matrix(activations[l, :]).T
    return (activation_T * deltas[l + 1, :]).ravel()


def gradient_update_matrix(activations, deltas):
    num_layers, d1 = activations.shape
    update_vals = ones([num_layers, d1 * (d1 - 1)])
    for l in range(num_layers):
        update_vals[l, :] = get_gradient_update_val(activations, deltas, l)
    return update_vals


def update_thetas(thetas, learningRate, gradient):
    thetas -= learningRate * gradient


def update_gradient(gradient, activations, deltas):
    gradient += gradient_update_matrix(activations, deltas)


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


def perturb(thetas, c):
    thetas_ravel = thetas.ravel()
    return thetas_ravel + c * eye(thetas_ravel.shape[0])


def get_grad_approx(c, cost_minus, cost_plus):
    return (cost_plus - cost_minus) / (2. * c)


def get_cost(X, y_bin, predictions, reg_factor, thetas):  # TODO implement
    no_reg = calculate_cost_no_reg(y_bin, predictions)
    reg = calculate_cost_reg(reg_factor, thetas, X.shape[0])
    return no_reg + reg
