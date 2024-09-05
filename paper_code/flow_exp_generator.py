"""
code for simple experiment generation
"""

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
import cvxpy as cp
import networkx as nx

@dataclass
class case_function:
    A: npt.ArrayLike
    x: npt.ArrayLike
    c: npt.ArrayLike
    cb: npt.ArrayLike
    ca: npt.ArrayLike
    def __post_init__(self):
        assert  self.cb.shape[0] == \
                self.ca.shape[0] == \
                self.A.shape[0]
        assert  self.c.shape[0] == \
                len(self.x) == \
                self.A.shape[1]
        self.cons = self.A @ self.x
    def get_func(self):
        return lambda b: self.cb @ b + self.ca @ cp.maximum(self.cons - b, 0.)
    def __call__(self, b, test = False):
        assert b.shape[0] == len(self.cb)
        if test:
            y = np.maximum(self.cons - b, 0.)
        else:
            y = cp.maximum(self.cons - b, 0.)
        result = self.cb @ b + self.ca @ y
        return result, y


def generate_cases(m: int, n: int, k: int):
    """
    function to generate case with m instances of problem,
    
    m -- number of instances
    n -- number of types of resourses
    k -- number of types of production
    
    consumption matrix of shape [n x k]
    x shape is [k]
    b, y shape is [n]
    """

    cases = []
    values = []
    b_s = []
    cb_base = 3 * (np.random.rand(n)* 1.8 + 1.)
    # ca = cb * (1.05 + np.random.rand(n) * 0.1)
    A = np.random.rand(n, k) + 0.2
    for _ in range(m):
        x = (np.random.rand(k)* 1. + 1.2) * 3

        cb = cb_base *(1 + np.random.rand()) * (0.3 + np.random.rand(n) * 0.5)
        ca = cb * (1.1 + np.random.rand(n) * 0.2)
        b_tmp = A @ x
        b = b_tmp * (0.95 + np.random.rand(n) * 0.1)    # пока что так, addition будет 0

        b_s.append(b)
        produce_cost = cb.T @ A  # production cost of products
        c = produce_cost *(1.05 + np.random.rand(k) * 0.2)
        
        cf = case_function(A, x, c, cb, ca)
        v = cf(b, True)[0]
        cases.append(cf)
        values.append(v)

    return cases, values, b_s