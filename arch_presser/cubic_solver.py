from typing import List
from scipy import optimize
import numpy as np

class CubicSolver:
    def __init__(self, left: float, right: float) -> None:
        self.left = left
        self.right = right
     
    def __call__(self, a: float, b: float, c: float, d: float)-> List[float]:
        """
            y = a * x^3 + b * x^2 + c * x + d
        """
        det = b * b - 3 * a * c
        def _fun(x: float) -> float:
            return ((a * x + b) * x + c) * x + d
        def _prime(x: float) -> float:
            return (3 * a * x + 2 * b) * x + c
        def _prime2(x: float) -> float:
            return 6 * a * x + 2 * b

            
        l = [self.left]
        
        if det > 0:
            polar_x0 = (-b - np.sqrt(det)) / (3 * a) 
            polar_x1 = (-b + np.sqrt(det)) / (3 * a)
            if self.left <= polar_x0 and polar_x0 <= self.right:
                l.append(polar_x0)
            if self.left <= polar_x1 and polar_x1 <= self.right:
                l.append(polar_x1)
        l.append(self.right)

        answers = []
        
        for i in range(len(l) - 1):
            if _fun(l[i]) * _fun(l[i + 1]) > 0:
                continue
            sol = optimize.root_scalar(_fun, bracket = (l[i], l[i + 1]), fprime = _prime,
                                        fprime2 = _prime2, x0 = (l[i] + l[ i + 1]) / 2)
            if sol.converged:
                answers.append(sol.root)
                
        return list(set(answers))