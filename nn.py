"""
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
"""
from numpy import random, zeros, ones, eye, matrix, apply_along_axis, unique
from numpy.core.umath import square
from numpy.ma import exp, true_divide, multiply, log
from numpy.testing import assert_almost_equal


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
    assert d1 == d + 1
    assert get_width(inputs) == d + 1
    inputs, theta = (matrix(m) for m in (inputs, theta))

    return sigmoid(inputs * theta)


def feed_forward(input, thetas, final_theta):
    d = get_width(input)
    activations = ones([thetas.shape[0] + 1, d + 1])
    # l (including output layer), d+1 (for bias nodes)
    for l, theta in enumerate(thetas):
        activations[l, 1:] = input
        assert activations[l, 0] == 1
        input = feed_forward_once(activations[l, :], reshape(theta, d))
    activations[-1, 1:] = input
    return activations, feed_forward_once(input, final_theta)  # = output


def feed_forward_multiple_inputs(inputs, thetas, final_theta):
    """
    :param inputs: matrix [[ n x d ]]
    :param thetas: matrix [[ num_layers x d(d+1) ]]
    :return: matrix [[ n x d ]]
    assumes num classes = d !
    """
    n, d = inputs.shape
    activations = ones([n, d + 1])
    for i, theta in enumerate(thetas):
        assert get_width(theta) == d * (d + 1)
        activations[:, 1:] = inputs
        inputs = feed_forward_once(activations, reshape(theta, d))
    return feed_forward_once(inputs, final_theta)  # = outputs


def get_error(output, classes, y_i):
    output[classes == y_i] = 1 - output[classes == y_i]
    return output


def reshape(theta, d):
    return theta.reshape(d + 1, d)


class NeuralNet:
    def __init__(self, layers, epsilon=0.12, learningRate=.1, numEpochs=100, gradientChecking=False):
        """
        Constructor
        Arguments:
        	layers - a numpy array of L-2 integers (L is # layers in the network)
        	epsilon - one half the interval around zero for setting the initial weights
        	learningRate - the learning rate for backpropagation
        	numEpochs - the number of epochs to run during training
        """
        self.layers = layers + 2  # includes input and output layer
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.numEpochs = numEpochs
        self.gradientChecking = gradientChecking

    def get_gradients(self, X, y):
        classes = unique(y)
        gradients = zeros(self.thetas.shape)
        for i, instance in enumerate(X):
            activations, output = feed_forward(instance, self.thetas, self.final_theta)
            g_prime = get_g_prime(activations)
            error = get_error(output, classes, y[i])
            deltas = get_deltas(g_prime, self.thetas, error)
            update_gradient(gradients, activations, deltas)
        return gradients

    def fit(self, X, y):
        """
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        """
        n, d = X.shape
        self.thetas = random.uniform(
            -self.epsilon, self.epsilon, size=(self.layers - 1, d + 1, d)
        )
        self.final_theta = random.uniform(
            -self.epsilon, self.epsilon, size=(d + 1, unique(y).size)
        )
        self.classes = unique(y)
        reg_factor = 1
        for _ in range(self.numEpochs):
            gradient = self.get_gradients(X, y)
            if self.gradientChecking:
                check_gradient(
                    gradient, X, y,
                    self.thetas,
                    self.final_theta,
                    self.classes,
                    reg_factor
                )
            regularize_gradient(gradient, reg_factor, self.thetas)
            update_thetas(self.thetas, self.learningRate, gradient)

    def predict(self, X):
        """
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        """
        return feed_forward_multiple_inputs(X, self.thetas, self.final_theta)

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
    theta_no_bias = reshape(thetas[l, :], get_width(deltas))[1:, :]
    return multiply(deltas[l + 1, :] * theta_no_bias.T, g_prime[l, :])


def get_deltas(g_prime, thetas, first_delta):
    """
    :param g_prime: sigma'(z), matrix [[ layers x d+1 ]]
    :param thetas: matrix [[ layers x (d x d+1) ]]
    :param first_delta: y - a^L, array [ d+1 ]
    :return: deltas, matrix [[ layers x d+1 ]]
    """
    num_layers, d = g_prime.shape
    deltas = zeros([num_layers + 1, d])
    deltas[-1, :] = first_delta
    hidden_layer_indices = range(num_layers - 1, 0, -1)
    for l in hidden_layer_indices:
        deltas[l, :] = next_delta(thetas, deltas, g_prime, l)
    return deltas


def get_gradient_update_val(activations, deltas, l):
    # rows ~ ouputs, columns ~ inputs
    return (activations[l, :].T * deltas[l + 1, :]).ravel()


def gradient_update_matrix(activations, deltas):
    num_layers, d1 = activations.shape
    update_vals = ones([num_layers, d1 * (d1 - 1)])
    for l in range(num_layers):
        update_vals[l, :] = get_gradient_update_val(activations, deltas, l)
    return update_vals


def update_thetas(thetas, final_theta, learningRate, gradient, final_gradient):
    thetas -= learningRate * gradient
    final_theta -= learningRate * final_gradient


def update_gradient(gradient, final_gradient, activations, deltas):
    gradient += gradient_update_matrix(activations, deltas)
    final_gradient += get_gradient_update_val()


def regularize_gradient(gradient, reg_factor, thetas):
    gradient[1:, :] += reg_factor * thetas


def get_cost(X, y, thetas, final_theta, classes, reg_factor):  # TODO implement
    predictions = feed_forward_multiple_inputs(X, thetas, final_theta)
    y_bin = get_y_bin(y, classes)
    no_reg = calculate_cost_no_reg(y_bin, predictions)
    reg = calculate_cost_reg(reg_factor, thetas, y.size)
    return no_reg + reg


def get_y_bin(y, classes):
    y_bin = zeros(y.size, classes.size)
    y_bin[classes == y] = 1
    return y_bin


def calculate_cost_no_reg(y_bin, predictions):
    n = y_bin.shape[0]
    term1 = y_bin * log(predictions).T
    term2 = (1 - y_bin) * log(1 - predictions).T
    return (term1 + term2) / n


def calculate_cost_reg(reg_factor, thetas, n):
    return reg_factor / (2 * n) * square(thetas).sum()


def perturb(thetas, c):
    return thetas + c * eye(thetas.shape[0])


def get_grad_approx(c, cost_minus, cost_plus):
    return (cost_plus - cost_minus) / (2 * c)


def recover(ravel_theta, thetas_shape, final_theta_shape):
    thetas_size = multiply(thetas_shape)
    thetas = ravel_theta[:thetas_size].reshape(thetas_shape)
    final_theta = ravel_theta[thetas_size:].reshape(final_theta_shape)
    return thetas, final_theta


def multicost(
        X, y, ravel_thetas,
        thetas_shape,
        final_theta_shape,
        classes,
        reg_factor
):
    def cost(ravel_theta):
        thetas, final_theta = recover(ravel_theta, thetas_shape, final_theta_shape)
        return get_cost(X, y, thetas, final_theta, classes, reg_factor)

    return apply_along_axis(cost, axis=1, arr=ravel_thetas)


def check_gradient(gradient, X, y, thetas, final_theta, classes, reg_factor):
    c = .1
    cost_plus, cost_minus = (multicost(
        X, y, perturb(thetas, c_),
        thetas.shape,
        final_theta.shape,
        classes,
        reg_factor
    ) for c_ in (c, -c))
    grad_approx = get_grad_approx(c, cost_minus, cost_plus)
    assert_almost_equal(gradient, grad_approx, decimal=4)
