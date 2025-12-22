# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np


def symlog(x: np.ndarray) -> np.ndarray:
    """Computes the symmetric logarithm of an array."""
    return np.sign(x) * np.log1p(abs(x))


def symexp(x: np.ndarray) -> np.ndarray:
    """Computes the symmetric exponential of an array."""
    return np.sign(x) * (np.exp(np.abs(x)) - 1)
