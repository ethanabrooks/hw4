from numpy import vstack

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
    theta = matrix([
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


def test_reshape():
    theta = matrix([1, 2, 3, 4, 5, 6])
    d = 2
    actual = reshape(theta, d)
    desired = matrix([
        [1, 2],
        [3, 4],
        [5, 6]
    ])
    assert_almost_equal(actual, desired)


def test_feed_forward():
    input = matrix([1, 2])
    thetas = zeros([0, 0])
    final_theta = matrix([
        [1, 2, 3, 4, 5, 6]
    ])
    activations = matrix([[1, 1, 2]])
    output = matrix([[0.9999991684719722, 0.9999999847700205]])
    desired = activations, output
    actual = feed_forward(input, thetas, final_theta)
    for a, d in zip(actual, desired):
        assert_almost_equal(a, d)

    thetas = matrix([
        [1, 2, 3, 4, 5, 6],
    ])
    final_theta = matrix([[11, 12, 13, 14, -15, -16]])
    actual = feed_forward(input, thetas, final_theta)
    activations2 = matrix([[1, 0.9999991684719722, 0.9999999847700205]])
    activations = vstack([activations, activations2])
    theta2 = matrix([
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
    thetas = zeros([0, 0])
    final_theta = random.uniform(-2, 2, size=(1, 6))
    outputs = []
    for input in inputs:
        outputs.append(feed_forward(input, thetas, final_theta)[1])
    desired = vstack(outputs)
    actual = feed_forward_multiple_inputs(inputs, thetas, final_theta)
    assert_almost_equal(desired, actual)


def test_get_g_prime():
    a = matrix([
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
    deltas = matrix([
        [0, 0],  # unused
        [2, -1]
    ])
    thetas = matrix([
        [4, 3, -3, 2, -5, -6]
    ])
    g_prime = matrix([
        [0, 5],
    ])
    desired = matrix([0, -20])
    actual = next_delta(thetas, deltas, g_prime, 0)
    # we would actually never calculate l=0
    assert_almost_equal(actual, desired)

    deltas = matrix([
        [0, 0],  # unused
        [-34.5, -60],
        [4, 5]
    ])
    thetas = matrix([
        [1, 2,  # unused
         4, -3, -5, 2],
        [11, 12,  # unused
         13, -15, 14, -16]
    ])
    g_prime = matrix([
        [.5, 2],
        [1.5, 2.5],
    ])
    desired = matrix([21, 105])
    actual = next_delta(thetas, deltas, g_prime, 0)
    # we would actually never calculate l=0
    assert_almost_equal(actual, desired)
    desired = matrix([-34.5, -60])
    actual = next_delta(thetas, deltas, g_prime, 1)
    assert_almost_equal(actual, desired)


def test_get_deltas():
    thetas = matrix([
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
    thetas = matrix([
        [1, 2, 4, -3, -5, 2],
    ])
    learning_rate = 1
    gradient = matrix([
        [11, 12, 0, 1, 0, 1]
    ])
    update_thetas(thetas, learning_rate, gradient)
    desired = matrix([-10, -10, 4, -4, -5, 1])
    assert_almost_equal(thetas, desired)


def test_calculate_cost():
    y = matrix([1, 2, 3])
    predictions = matrix([1, 1, 4])
    desired = matrix([0, ])
