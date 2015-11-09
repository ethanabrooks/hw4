"""
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
"""
from PIL import Image
from numpy import ma, c_, dot, array
from numpy import random, zeros, ones, matrix, unique, argmax, fromfunction, vectorize, where, r_
from numpy.core.umath import square
from numpy.ma import multiply, log, floor, sqrt
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
    dot_product = ma.dot(inputs, theta, strict=True)
    return expit(dot_product)


def init_thetas(epsilon, hidden_layers, hl_size, num_features, num_classes, rand=True):
    size = hidden_layers, (hl_size + 1) * hl_size
    theta1_width = num_classes if hidden_layers == 0 else hl_size
    theta1size = num_features + 1, theta1_width
    # if hidden_layers == 0:
    #     size = (num_features, num_classes)
    #     if rand:
    #         return random.uniform(-epsilon, epsilon, size=size)
    #     else:
    #         return ones(size)
    if rand:
        theta1 = random.uniform(-epsilon, epsilon, size=(theta1size))
        thetas_unmasked = random.uniform(-epsilon, epsilon, size=size)
    else:
        theta1 = ones(theta1size)
        thetas_unmasked = ones(size)
    mask = zeros(thetas_unmasked.shape)
    mask[-1].reshape(hl_size + 1, hl_size)[:, num_classes:] = 1
    return theta1, ma.array(data=thetas_unmasked, mask=mask, fill_value=0)


def feed_forward(input, theta1, thetas, K=None):
    activations = ma.ones([thetas.shape[0], theta1.shape[1] + 1])
    # L (including output layer), d+1 (for bias nodes)
    d = None
    input = matrix(c_[1, input]) * theta1
    activations[0, 1:] = input
    for l, theta in enumerate(thetas):
        activations[l, 1:] = input
        assert activations[l, 0] == 1
        if l + 1 == thetas.shape[0]:
            d = K  # on the final layer, we shorten the width of theta
        theta_ = reshape(theta, d)
        input = feed_forward_once(activations[l, :], theta_)
    return activations, input  # = output


def feed_forward_multiple_inputs(inputs, theta1, thetas, K=None):
    """
    :param theta1:
    :param inputs: matrix [[ n x d ]]
    :param thetas: matrix [[ num_layers x d(d+1) ]]
    :return: matrix [[ n x d ]]
    assumes num classes = d !
    """
    n, d = inputs.shape
    inputs = dot(c_[ones((n, 1)), inputs], theta1)
    d = None
    for l, theta in enumerate(thetas):
        activations = c_[ones((n, 1)), inputs]
        if l + 1 == thetas.shape[0]:
            d = K  # on the final layer, we shorten the width of theta
        theta_ = reshape(theta, d)
        inputs = feed_forward_once(activations, theta_)
    return inputs  # = outputs


def get_error(output, classes, y_i):
    error = output.copy()
    error[where(classes == y_i)] = output[where(classes == y_i)] - 1
    return error


def reshape(theta, d=None):
    d1 = floor(sqrt(theta.size)) + 1
    theta_ = theta.reshape(d1, -1)
    return theta_ if d is None else theta_[:, :d]


class NeuralNet:
    def __init__(self, layers, epsilon=0.12, learningRate=.5, numEpochs=100, gradientChecking=False, randTheta=True, hl_size=25):
        """
        Constructor
        Arguments:
        	layers - a numpy array of L-2 integers (L is # layers in the network)
        	epsilon - one half the interval around zero for setting the initial weights
        	learningRate - the learning rate for backpropagation
        	numEpochs - the number of epochs to run during training
        """
        self.hidden_layers = layers
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.numEpochs = numEpochs
        self.gradientChecking = gradientChecking
        self.randTheta = randTheta
        self.hl_size = hl_size
        self.reg_factor = .0001

    def get_gradients(self, X, y):
        gradient1 = zeros(self.theta1.shape)
        gradients = ma.zeros(self.thetas.shape)
        gradients.mask = self.thetas.mask
        for i, instance in enumerate(X):
            activations, output = feed_forward(instance, self.theta1, self.thetas, K=self.classes.size)
            g_prime = get_g_prime(activations)
            error = get_error(output, self.classes, y[i])
            deltas = get_deltas(g_prime, self.thetas, error)
            update_gradients(gradient1, gradients, instance, activations, deltas)
        return gradient1 / X.shape[0], gradients / X.shape[0]

    def multicost(self, X, y_bin, thetas, c):
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
            predictions = feed_forward_multiple_inputs(X, self.theta1, perturbed_thetas)
            return get_cost(
                X, y_bin, predictions[:, :self.classes.size], reg_factor, perturbed_thetas
            )

        reg_factor = self.reg_factor
        costs = fromfunction(vectorize(cost), thetas.shape)
        return ma.array(data=costs, mask=thetas.mask)

    def check_gradient(self, gradient, X, y, c=.0001):
        y_bin = get_y_bin(y, self.classes)
        cost_plus, cost_minus = (self.multicost(
            X, y_bin, self.thetas, c_,
        ) for c_ in (c, -c))
        grad_approx = get_grad_approx(c, cost_minus, cost_plus)
        assert_almost_equal(gradient, grad_approx, decimal=4)

    def fit(self, X, y):
        """
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        """
        X = matrix(X)
        self.classes = unique(y)
        self.theta1, self.thetas = init_thetas(
            self.epsilon, self.hidden_layers, self.hl_size,
            X.shape[1], self.classes.size, rand=self.randTheta
        )
        for _ in range(self.numEpochs):
            gradient1, gradient = self.get_gradients(X, y)
            regularize_gradient(gradient1, gradient, self.reg_factor, self.theta1, self.thetas)
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
        return feed_forward_multiple_inputs(X, self.theta1, self.thetas)

    def score(self, X_train, y_train, X_test, y_test):
        self.fit(X_train, y_train)
        probabilities = self.predict(X_test)
        y_pred = self.classes[argmax(probabilities, axis=1)]
        return accuracy_score(y_test, y_pred)

    def visualizeHiddenNodes(self, filename):
        """
        CIS 519 ONLY - outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        """

        big_img = Image.new('RGB', (100, 100), "black")  # create a new black image
        for weight in self.theta1:
            weights = weight.reshape(20, 20)
            sml_img = Image.new('RGB', (20, 20), "black")
            pixels = sml_img.load()
            for i in range(sml_img.size[0]):  # for every pixel:
                for j in range(sml_img.size[1]):
                    color = weights[i, j]
                    pixels[i, j] = color
                    
        img.show()


def get_g_prime(activations):
    return multiply(activations, 1 - activations)[:, 1:]  # omit bias activations


def next_delta(thetas, deltas, g_prime, l):
    #  recover weights for lth layer, omitting bias nodes
    theta_no_bias = reshape(thetas[l, :])[1:, :]
    return multiply(ma.dot(deltas[l + 1, :], theta_no_bias.T), g_prime[l, :])


def get_deltas(g_prime, thetas, last_delta):
    """
    layers does not include output
    :param g_prime: sigma'(z), matrix [[ layers x d+1 ]]
    :param thetas: matrix [[ layers x (d x d+1) ]]
    :param last_delta: y - a^L, array [ d+1 ]
    :return: deltas, matrix [[ layers x d+1 ]]
    """
    num_layers, d = g_prime.shape
    deltas = ma.zeros([num_layers + 1, d])  # true num layers
    deltas[-1, :last_delta.size] = last_delta
    hidden_layer_indices = range(num_layers - 1, 0, -1)
    for l in hidden_layer_indices:
        deltas[l, :] = next_delta(thetas, deltas, g_prime, l)
    return deltas


def get_gradient_update_val(activations, delta):
    # rows ~ ouputs, columns ~ inputs
    activation_T = matrix(activations).T
    dot_product = activation_T * delta
    return dot_product.ravel()


def gradient_update_matrix(activations, deltas):
    num_hl, d1 = activations.shape
    update_vals = ma.ones([num_hl, d1 * (d1 - 1)])
    for l in range(num_hl):
        update_vals[l, :] = get_gradient_update_val(
            activations[l, :], deltas[l + 1, :]
        )
    return update_vals


def update_thetas(thetas, learningRate, gradient):
    thetas -= learningRate * gradient


def update_gradients(gradient1, gradient, input, activations, deltas):
    # input needs to be a 1 x d matrix
    gradient1_update = get_gradient_update_val(c_[1, input], deltas[1, :])
    update_matrix = gradient_update_matrix(activations, deltas)
    gradient1 += gradient1_update.reshape(gradient1.shape)
    gradient += update_matrix


def regularize_gradient(gradient1, gradient, reg_factor, theta1, thetas):
    gradient1 += reg_factor * theta1
    gradient += reg_factor * thetas


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
