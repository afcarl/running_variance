import numpy as np


class RunningVariance(object):

    def __init__(self):

        self._mean = 0.0
        self._var = 0.0

        self._num_observations = 0

    def push(self, x):

        self._num_observations += 1

        if self._num_observations == 1:
            self._mean = x
            return

        new_mean = self._mean + (x - self._mean) / self._num_observations
        new_var = self._var + (x - self._mean) * (x - new_mean)

        self._mean = new_mean
        self._var = new_var

    def pop(self, x):

        if self._num_observations <= 1:
            self._mean = 0.0
            self._num_observations -= 1
            return

        new_mean = ((self._mean * self._num_observations - x)
                    / (self._num_observations - 1))
        new_var = self._var - (x - new_mean) * (x - self._mean)

        self._num_observations -= 1

        self._mean = new_mean
        self._var = new_var

    def var(self):

        if self._num_observations > 1:
            return (self._var / (self._num_observations - 1))
        else:
            return 0.0

    def mean(self):

        return self._mean


def test_mean_and_variance():

    mean = 4.5
    variance = 1.0
    num_observations = 1000

    data = np.random.normal(mean, variance, size=num_observations)

    measurement = RunningVariance()

    for i, x in enumerate(data):
        measurement.push(x)

        if i > 0:
            assert np.allclose(measurement.mean(), np.mean(data[:(i + 1)]))
            assert np.allclose(measurement.var(), np.var(data[:(i + 1)], ddof=1))

    for i, x in enumerate(data):

        measurement.pop(x)

        if i < 998:
            assert np.allclose(measurement.mean(), np.mean(data[i + 1:]))
            assert np.allclose(measurement.var(), np.var(data[i + 1:], ddof=1))
