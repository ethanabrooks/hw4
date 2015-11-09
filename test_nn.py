from numpy import vstack, hstack, array, c_, r_
from numpy.ma import array

__author__ = 'Ethan'
from nn import *


def test_feed_forward_once():
    inputs = matrix([
        [1, 1, 2],
        [4, 3, 1]
    ])
    theta = array([
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
    num_hl = 2
    num_features = 5
    hl_width = 3
    num_classes = 1
    desired = ones(46), array([
        [6, 3],
        [4, 3],
        [4, 3],
        [4, 1],
    ])
    actual = init_thetas(1, num_hl, num_features, hl_width, num_classes, rand=False)
    for a, d in zip(actual, desired):
        assert_almost_equal(a, d)


# def test_reshape():
#     theta = array([1, 2, 3, 4, 5, 6])
#     actual = recover(theta)
#     desired = matrix([
#         [1, 2],
#         [3, 4],
#         [5, 6]
#     ])
#     assert_almost_equal(actual, desired)


def test_feed_forward():
    input = matrix([1, 2])
    thetas = array([
        [1, 2, 3, 4, 5, 6]
    ])
    thetas_shapes = matrix([[3, 2]])
    activations = array([1, 1, 2])
    output = array([0.9999991684719722, 0.9999999847700205])
    desired = activations, matrix([3]), output
    actual = feed_forward(input, thetas, thetas_shapes)
    for a, d in zip(actual, desired):
        assert_almost_equal(a, d)

    thetas = array([
        [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, -15, -16]
    ])
    thetas_shapes = array([
        [3, 2],
        [3, 2]
    ])
    actual = feed_forward(input, thetas, thetas_shapes)
    activations2 = array([1, 0.9999991684719722, 0.9999999847700205])
    activations = array(r_[activations, activations2])
    theta2 = array([
        [11, 12],
        [13, 14],
        [-15, -16]
    ])
    output = feed_forward_once(activations2.flatten(), theta2)
    desired = activations, array([3, 3]), output
    for a, d in zip(actual, desired):
        assert_almost_equal(a, d)


def test_feed_forward_multiple_inputs():
    inputs = random.uniform(-2, 2, size=(2, 2))
    thetas = array(random.uniform(-2, 2, size=(1, 6)))
    thetas_shapes = array([[3, 2]])
    outputs = []
    for input in inputs:
        outputs.append(feed_forward(input, thetas, thetas_shapes)[2])
    desired = vstack(outputs)
    actual = feed_forward_multiple_inputs(inputs, thetas, thetas_shapes)
    assert_almost_equal(desired, actual)


def test_get_g_prime():
    a = array([
        [1, 1, 2, 1, 3, 4]
    ])
    desired = [
        array([0, -2]),
        array([-6, -12])
    ]
    actual = get_g_prime(a, array([3, 3]))
    for a, d in zip(actual, desired):
        assert_almost_equal(a, d)


def test_next_delta():
    deltas = [array([2, -1])]
    thetas = [array([[4, 3], [-3, 2], [-5, -6]])]
    g_prime = [array([[0, 5]])]
    desired = array([[0, -20]])
    actual = next_delta(thetas, deltas, g_prime, 0)
    # we would actually never calculate l=0
    assert_almost_equal(actual, desired)

    deltas = [array([[4, 5]])]
    thetas_list = [
        array([[1, 2],  # unused
               [4, -3], [-5, 2]]),
        array([[11, 12],  # unused
               [13, -15], [14, -16]])
    ]
    g_prime = [
        array([[.5, 2]]),
        array([[1.5, 2.5]])
    ]
    desired = array([[.5, -20]])
    actual = next_delta(thetas_list, deltas, g_prime, 0)
    # we would actually never calculate l=0

    deltas = [array([0, -20])]
    thetas_list = [
        array([[1, 1],  # unused
               [6, -1], [4, -1]])
    ]
    g_prime = [
        array([[2, .5]])
    ]
    assert_almost_equal(actual, desired)
    desired = array([[40, 10]])
    actual = next_delta(thetas_list, deltas, g_prime, 0)
    assert_almost_equal(actual, desired)


def test_get_deltas():
    thetas = [
        array([[1, 1], [6, -1], [4, -1]]),  # 1st hidden
        array([[4, 3], [-3, 2], [-5, -6]])  # input to 1st hidden
    ]
    g_prime = [
        # matrix([2, .5]),  # intput (not used)
        matrix([2, .5]),  # hidden
        matrix([0, 5])  # output
    ]
    first_delta = matrix([2, -1])
    actual = get_deltas(g_prime, thetas, first_delta)
    desired = [
        matrix([40, 10]),
        matrix([0, -20]),
        matrix([2, -1])
    ]
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
    actual = gradient_update_vector(activations, deltas)
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
    actual = gradient_update_vector(activations, deltas)
    desired = matrix([
        [3, -3, 6, -6, 9, -9],
        [-1, 1, -2, 2, -3, 3]
    ])
    assert_almost_equal(actual, desired)


def test_update_theta():
    thetas = array([
        [1, 2, 4, -3, -5, 2],
    ])
    learning_rate = 1
    gradient = array([
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
    theta = array([
        [1, -2],
        [1, 1],
        [-3, 4]
    ])
    n = 1
    desired = 1 + 4 + 1 + 1 + 9 + 16
    actual = calculate_cost_reg(reg_factor, theta, n)
    assert_almost_equal(actual, desired)

    reg_factor = 1
    theta = array([
        [1, -2],
        [1, 1],
        [-3, 4]
    ])
    n = 2
    desired = 8
    actual = calculate_cost_reg(reg_factor, theta, n)
    assert_almost_equal(actual, desired)


# def test_perturb():
#     theta = array([
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
    cost_minus = array([
        [0, -3],
        [-3, 4]
    ])
    cost_plus = array([
        [1, -1],
        [-3, 4]
    ])
    desired = array([
        [.5, 1],
        [0, 0]
    ])
    actual = get_grad_approx(c, cost_minus, cost_plus)
    assert_almost_equal(actual, desired)


def test_check_gradient():
    net = NeuralNet(0, gradientChecking=True, randTheta=False)

    X = matrix([
        [1],
    ])
    y = array([1])
    net.fit(X, y)

    n = 10
    X = random.rand(n, 4)
    y = random.randint(0, 3, n)
    # net.fit(X, y)

    X = matrix([
        [1, 1, 0, 0, 1],
        [0, 0, 1, 0, 0]
    ])
    y = array([0, 0])
    # net.fit(X, y)
