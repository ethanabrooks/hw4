from numpy import vstack, hstack, array
from numpy.ma import masked_array

__author__ = 'Ethan'
from nn import *


def test_sigmoid():
    x = .5
    actual = sigmoid(x)
    desired = 0.6224593
    assert_almost_equal(actual, desired)

    x = matrix([
        [1, 2],
        [3, 4]
    ])
    desired = matrix([
        [0.7310585786300049, 0.8807970779778823],
        [0.9525741268224334, 0.9820137900379085]
    ])
    actual = sigmoid(x)
    assert_almost_equal(actual, desired)


def test_feed_forward_once():
    inputs = matrix([
        [1, 1, 2],
        [4, 3, 1]
    ])
    theta = masked_array([
        [1, -2],
        [1.2, 1.6],
        [-3, 4]
    ])
    desired = matrix([
        [0.021881270936130476, 0.9994997988929205],
        [0.9900481981330957, 0.6899744811276125]
    ])
    actual = feed_forward_once(inputs, theta)
    assert_almost_equal(actual, desired)

def test_init_thetas():
    d = 2
    layers = 1
    num_classes = 2
    desired = masked_array(ones((1, 6)))
    actual = init_thetas(1, layers, d, num_classes, False)
    assert_almost_equal(actual, desired)



def test_reshape():
    theta = masked_array([1, 2, 3, 4, 5, 6])
    actual = reshape(theta)
    desired = matrix([
        [1, 2],
        [3, 4],
        [5, 6]
    ])
    assert_almost_equal(actual, desired)


def test_feed_forward():
    input = matrix([1, 2])
    thetas = masked_array([
        [1, 2, 3, 4, 5, 6]
    ])
    activations = masked_array([[1, 1, 2]])
    output = masked_array([0.9999991684719722, 0.9999999847700205])
    desired = activations, output
    actual = feed_forward(input, thetas)
    for a, d in zip(actual, desired):
        assert_almost_equal(a, d)

    thetas = masked_array([
        [1, 2, 3, 4, 5, 6],
        [11, 12, 13, 14, -15, -16]
    ])
    actual = feed_forward(input, thetas)
    activations2 = masked_array([[1, 0.9999991684719722, 0.9999999847700205]])
    activations = masked_array(vstack([activations, activations2]))
    theta2 = masked_array([
        [11, 12],
        [13, 14],
        [-15, -16]
    ])
    output = feed_forward_once(activations2.flatten(), theta2)
    desired = activations, output
    for a, d in zip(actual, desired):
        assert_almost_equal(a, d)


def test_feed_forward_multiple_inputs():
    inputs = random.uniform(-2, 2, size=(2, 2))
    thetas = masked_array(random.uniform(-2, 2, size=(1, 6)))
    outputs = []
    for input in inputs:
        outputs.append(feed_forward(input, thetas)[1])
    desired = vstack(outputs)
    actual = feed_forward_multiple_inputs(inputs, thetas)
    assert_almost_equal(desired, actual)


def test_get_g_prime():
    a = masked_array([
        [1, 1, 2],
        [1, 3, 4]
    ])
    desired = matrix([
        [0, -2],
        [-6, -12]
    ])
    actual = get_g_prime(a)
    assert_almost_equal(actual, desired)


def test_next_delta():
    deltas = masked_array([
        [0, 0],  # unused
        [2, -1]
    ])
    thetas = masked_array([
        [4, 3, -3, 2, -5, -6]
    ])
    g_prime = masked_array([
        [0, 5],
    ])
    desired = masked_array([0, -20])
    actual = next_delta(thetas, deltas, g_prime, 0)
    # we would actually never calculate l=0
    assert_almost_equal(actual, desired)

    deltas = masked_array([
        [0, 0],  # unused
        [-34.5, -60],
        [4, 5]
    ])
    thetas = masked_array([
        [1, 2,  # unused
         4, -3, -5, 2],
        [11, 12,  # unused
         13, -15, 14, -16]
    ])
    g_prime = masked_array([
        [.5, 2],
        [1.5, 2.5],
    ])
    desired = masked_array([21, 105])
    actual = next_delta(thetas, deltas, g_prime, 0)
    # we would actually never calculate l=0
    assert_almost_equal(actual, desired)
    desired = masked_array([-34.5, -60])
    actual = next_delta(thetas, deltas, g_prime, 1)
    assert_almost_equal(actual, desired)


def test_get_deltas():
    thetas = masked_array([
        [1, 2, 4, -3, -5, 2],
        [11, 12, 13, -15, 14, -16],
        [11, 12, 0, 1, 0, 1]
    ])
    g_prime = matrix([
        [.5, 2],
        [1.5, 2.5],
        [4, 5],
    ])
    first_delta = matrix([1, 1])
    actual = get_deltas(g_prime, thetas, first_delta)
    desired = matrix([
        [0, 0],
        [-34.5, -60],
        [4, 5],
        [1, 1]
    ])
    assert_almost_equal(actual, desired)


def test_get_gradient_update_val():
    activations = matrix([
        [1, 2, 3]
    ])
    deltas = matrix([
        [0, 0],
        [3, -3]
    ])
    actual = get_gradient_update_val(activations, deltas, 0)
    desired = matrix([3, -3, 6, -6, 9, -9])
    assert_almost_equal(actual, desired)


def test_gradient_update_matrix():
    activations = matrix([
        [1, 2, 3]
    ])
    deltas = matrix([
        [0, 0],
        [3, -3]
    ])
    actual = gradient_update_matrix(activations, deltas)
    desired = matrix([[3, -3, 6, -6, 9, -9]])
    assert_almost_equal(actual, desired)

    activations = matrix([
        [1, 2, 3],
        [-1, -2, -3]
    ])
    deltas = matrix([
        [0, 0],
        [3, -3],
        [1, -1]
    ])
    actual = gradient_update_matrix(activations, deltas)
    desired = matrix([
        [3, -3, 6, -6, 9, -9],
        [-1, 1, -2, 2, -3, 3]
    ])
    assert_almost_equal(actual, desired)


def test_update_theta():
    thetas = masked_array([
        [1, 2, 4, -3, -5, 2],
    ])
    learning_rate = 1
    gradient = masked_array([
        [11, 12, 0, 1, 0, 1]
    ])
    update_thetas(thetas, learning_rate, gradient)
    desired = matrix([-10, -10, 4, -4, -5, 1])
    assert_almost_equal(thetas, desired)


def test_get_error():
    output = array([.1, .2, .7])
    classes = array(['red', 'green', 'blue'])
    y_i = 'blue'
    actual = get_error(output, classes, y_i)
    desired = [.1, .2, -.3]
    assert_almost_equal(actual, desired)


def test_get_y_bin():
    classes = array(['red', 'green'])
    y = array(['red', 'red', 'green'])
    actual = get_y_bin(y, classes)
    desired = array([
        [1, 0],
        [1, 0],
        [0, 1]
    ])
    assert_almost_equal(actual, desired)


def test_calculate_cost_no_reg():
    y = matrix([1, 0, 0])
    predictions = matrix([.1, .2, .7])
    desired = 3.72970145
    actual = calculate_cost_no_reg(y, predictions)
    assert_almost_equal(actual, desired)

    y = matrix([0, 0, 1])
    predictions = matrix([.1, .2, .7])
    actual = calculate_cost_no_reg(y, predictions)
    desired = 0.68517901
    assert_almost_equal(actual, desired)

    y = array([
        [1, 0, 0],
        [0, 0, 1]
    ])
    predictions = array([
        [.1, .2, .7],
        [.1, .2, .7]
    ])
    actual = calculate_cost_no_reg(y, predictions)
    desired = 2.20744023
    assert_almost_equal(actual, desired)


def test_calculate_cost_reg():
    reg_factor = 2
    theta = masked_array([
        [1, -2],
        [1, 1],
        [-3, 4]
    ])
    n = 1
    desired = 1 + 4 + 1 + 1 + 9 + 16
    actual = calculate_cost_reg(reg_factor, theta, n)
    assert_almost_equal(actual, desired)

    reg_factor = 1
    theta = masked_array([
        [1, -2],
        [1, 1],
        [-3, 4]
    ])
    n = 2
    desired = 8
    actual = calculate_cost_reg(reg_factor, theta, n)
    assert_almost_equal(actual, desired)


# def test_perturb():
#     theta = masked_array([
#         [1, -2],
#         [1, 1],
#     ])
#     c = 1
#     desired = array([
#         [2, -2, 1, 1],
#         [1, -1, 1, 1],
#         [1, -2, 2, 1],
#         [1, -2, 1, 2]
#     ])
#     actual = perturb(theta, c)
#     assert_almost_equal(actual, desired)


def test_get_grad_approx():
    c = 1
    cost_minus = masked_array([
        [0, -3],
        [-3, 4]
    ])
    cost_plus = masked_array([
        [1, -1],
        [-3, 4]
    ])
    desired = masked_array([
        [.5, 1],
        [0, 0]
    ])
    actual = get_grad_approx(c, cost_minus, cost_plus)
    assert_almost_equal(actual, desired)


def test_check_gradient():
    net = NeuralNet(1, gradientChecking=True, randTheta=False)
    X = matrix([
        [1],
        [1]
    ])
    y = array([1, 1])
    net.fit(X, y)

    X = matrix([
        [1, 0],
        [0, 1]
    ])
    y = array([0, 1])
    net.fit(X, y)
