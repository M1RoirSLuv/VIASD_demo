"""Shared numpy utilities for softmax / log-softmax."""

import numpy as np


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    s = x - np.max(x, axis=axis, keepdims=True)
    return s - np.log(np.sum(np.exp(s), axis=axis, keepdims=True))
