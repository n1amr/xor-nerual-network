"""
Author: Amr Alaa
"""

# Python 2
from __future__ import division
from __future__ import absolute_import
import sys

import numpy as np
from matplotlib import pyplot
from numpy import (array, exp, matmul, reshape, negative, concatenate, outer, zeros, ones)
from numpy.random import rand

if sys.version_info <= (3, 0):
    print('WARNING: this script is written in Python 3.5.1 please use python 3.x')
    from itertools import izip

    range = xrange
    zip = izip
    input = raw_input


def sigmoid(x):
    return 1 / (1 + exp(negative(x)))


def linear(x):
    return x


plot_error = False


def init_weights(num_neurons, epsilon):
    """Returns matrices of random elements in range(-epsilon, epsilon)
    matrix #0 denotes the weights matrix from layer 0 (input layer) to
    layer 1 (first hidden layer). and so on. Note that bias term is the first column
    """
    num_layers = len(num_neurons)
    weights = [None] * (num_layers - 1)
    for layer in range(num_layers - 1):
        p = num_neurons[layer]
        n = num_neurons[layer + 1]
        weights[layer] = rand(n, p + 1) * 2 * epsilon - epsilon
    return weights


def copy_weights(weights):
    return [w.copy() for w in weights]


def predict(x_input, num_neurons, weights, f_act_out):
    """ Applies weights on the input set and returns the outputs of the output layer
    """
    out = forward(x_input, num_neurons, weights, f_act_out)
    return np.int32(np.round(out[-1][:, 1:]))


def score(x_train, y_train, num_neurons, weights, f_act_out):
    """Calculates the score of the weights which is the number of
    correctly predicted outputs using this weights matrix
    """
    y_predicted = predict(x_train, num_neurons, weights, f_act_out)
    return sum(1 for eq_row in y_predicted == y_train if all(eq_row))


def forward(x_train, num_neurons, weights, f_act_out):
    """Forward propagation.

    returns list of vectors that contains the output of neurons in each layer.
    Note that the first element of each vector is a constant value of 1
    to simplify calculations for bias
    """
    num_layers = len(num_neurons)
    num_samples = len(x_train)

    out = [zeros([num_samples, num_neurons[layer] + 1]) for layer in range(num_layers)]

    out[0] = concatenate([ones([num_samples, 1]), x_train], 1)
    for layer in range(1, num_layers):
        in_ = out[layer - 1]
        w = weights[layer - 1]
        net = np.dot(w, in_.T).T
        out_ = sigmoid(net) if layer < num_layers - 1 else f_act_out(net)
        out[layer] = concatenate([ones([num_samples, 1]), out_], 1)

    return out


def train(x_train, y_train, num_neurons, epsilon=0.12, learning_rate=0.05, max_iter=30000, momentum=0.95,
          f_act_out=sigmoid):
    """ Back propagation training
    """
    check_after_iterations = 10000

    weights = init_weights(num_neurons, epsilon)
    optimal_weights = copy_weights(weights)
    last_adaptation = [w * 0 for w in weights]
    max_score = 0

    num_layers = len(num_neurons)
    num_samples = len(x_train)

    y_train = reshape(y_train, [num_samples, -1])

    delta = []
    errors = []

    for iteration in range(max_iter):
        out = forward(x_train, num_neurons, weights, f_act_out)

        delta = [zeros([num_samples, num_neurons[layer]]) for layer in range(num_layers)]
        for layer in range(num_layers - 1, 0, -1):
            out_u = out[layer][:, 1:]
            lmd = out_u * (1 - out_u)

            out_p = out[layer - 1]

            if layer != num_layers - 1:
                # hidden layers
                w_s_u = weights[layer]  # from layer to layer + 1
                delta_s = delta[layer + 1]

                delta_u = np.dot(w_s_u.T, delta_s.T).T
                delta_u = delta_u[:, 1:] * lmd
            else:
                # output layer
                if f_act_out == sigmoid:
                    delta_u = (y_train - out_u) * lmd
                elif f_act_out == linear:
                    delta_u = (y_train - out_u)

            delta[layer] = delta_u

            w_u_p = weights[layer - 1]
            adaptation = momentum * last_adaptation[layer - 1] + learning_rate * sum(
                outer(delta_u[sample], out_p[sample]) for sample in range(num_samples))
            w_u_p += adaptation
            last_adaptation[layer - 1] = adaptation
            weights[layer - 1] = w_u_p

        d = delta[-1][:, -1]
        error = d.dot(d)
        errors.append(error)

        if iteration % check_after_iterations == 0:
            new_score = score(x_train, y_train, num_neurons, weights, f_act_out)

            if new_score > max_score:
                max_score, optimal_weights = new_score, copy_weights(weights)
            if iteration % check_after_iterations == 0 and iteration > 0:
                print('completed {0} iterations: current score = {1}, '
                      'max score = {2}'.format(iteration, new_score, max_score))
                if new_score < max_score:
                    break
            if new_score == len(x_train):
                break  # 100% score

    d = delta[-1][:, -1]
    error = d.dot(d)
    print('max score = {0}, error = {1}'.format(max_score, error))

    if plot_error:
        pyplot.plot(errors)
        pyplot.show()

    return optimal_weights


def run(x_train, y_train, num_neurons, f_act_out, num_trials=3):
    max_score = -1
    optimal_weights = None

    for i in range(num_trials):
        print('\ntrial {0:02}'.format(i + 1))
        print('=' * 10)
        new_weights = train(x_train, y_train, num_neurons, f_act_out=f_act_out)
        new_score = score(x_train, y_train, num_neurons, new_weights, f_act_out)
        if new_score > max_score:
            max_score = new_score
            optimal_weights = new_weights
        print('-' * 80)
        print('trial {0:02}: score: {1}, max score: {2} ({3}%)'.format(i + 1, new_score, max_score,
                                                                       max_score / len(x_train) * 100))

        if new_score == len(x_train):
            break

    print('=' * 80)
    print('\n')
    print('optimal weights:')
    for i, w in enumerate(optimal_weights):
        print('From layer {0} to layer {1}:'.format(i, i + 1))
        print(w)
        print('\n')
    print("max score = {0}/{1} ({2}%)".format(max_score, len(x_train), max_score / len(x_train) * 100))

    y_predicted = predict(x_train, num_neurons, optimal_weights, f_act_out)
                                for x, y, y2 in zip(x_train, y_predicted, y_train):
        print('{0} -> {1}: {2}'.format(x, y, all(y == y2)))

    return optimal_weights


def main():
    s = input('plot errors? [y/n] ')
    if s.strip().count('y') > 0:
        global plot_error
        plot_error = True
    summary = []

    for f_act_out in [sigmoid, linear, ]:
        print('\n')
        print('=' * 80)
        print('training 3-input xor')
        summary.append('using {0} activation function'.format(f_act_out.__name__))
        print(summary[-1])

        x_train = array([(x1, x2, x3) for x1 in range(2) for x2 in range(2) for x3 in range(2)])
        y_train = array([[sum(x) % 2] for x in x_train])
        numbers = list(range(1, 9))
        scores = [0 for _ in numbers]

        for i in range(len(numbers)):
            print('\nusing {0} hidden neurons'.format(numbers[i]))
            num_neurons = [len(x_train[0]), numbers[i], len(y_train[0])]
            weights = run(x_train, y_train, num_neurons, f_act_out)
            s = score(x_train, y_train, num_neurons, weights, f_act_out)
            scores[i] = s / len(x_train) * 100
            print('#' * 80)

        # pyplot.plot(numbers, scores)
        # pyplot.ylim([0, 100])
        # pyplot.show()

        min_index = len(scores) - 1
        while min_index > 0 and scores[min_index - 1] == max(scores):
            min_index -= 1
        min_number = numbers[min_index]

        summary.append(', '.join('{0} -> {1}%'.format(n, s) for (n, s) in zip(numbers, scores)))
        print(summary[-1])
        summary.append('minimum number of neurons that makes best score is {0}'.format(min_number))
        print(summary[-1])
        summary.append('')

    print('=' * 80)
    print('summary')
    print('=' * 10)
    print('\n'.join(summary))


if __name__ == '__main__':
    main()
