# This file contains utility functions for this project.
# Author: Qidong Yang & Jonathan Giezendanner
# Date: 2024-02-14


class MinMaxNormalizer(object):
    def __init__(self, min, max, eps=0.00001):
        super(MinMaxNormalizer, self).__init__()

        # normalization using overall maximum and minmum so falling into [0, 1]

        self.max = max
        self.min = min
        self.eps = eps

    def encode(self, x):
        x = (x - self.min) / (self.max - self.min + self.eps)

        return x

    def decode(self, x):
        x = x * (self.max - self.min + self.eps) + self.min

        return x


class ABNormalizer(object):
    def __init__(self, min_x, max_x, a, b, eps=0.00001):
        super(ABNormalizer, self).__init__()

        # normalization using overall maximum and minimum, rescaling between a and b, so falling into [a, b]

        self.max = max_x
        self.min = min_x
        self.a = a
        self.b = b
        self.eps = eps
        self.delta = (self.b - self.a) / (self.max - self.min + self.eps)

    def encode(self, x):
        x = (x - self.min) * self.delta + self.a

        return x

    def decode(self, x):
        x = (x - self.a) / self.delta + self.min

        return x
