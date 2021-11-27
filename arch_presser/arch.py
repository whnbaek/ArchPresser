from typing import List, Tuple, Union
import numpy as np
import scipy.integrate as integrate

from .cubic_solver import CubicSolver

class Arch:
    epsilon = 1e-3
    def __init__(self, upper_start_points: List[Tuple[float, float, float]],
                 upper_end_points: List[Tuple[float, float, float]],
                 lower_start_points: List[Tuple[float, float, float]],
                 lower_end_points: List[Tuple[float, float, float]],
                 h: float, w: float) -> None:
        assert len(upper_start_points + lower_start_points) >= 3 and len(upper_end_points + lower_end_points) >= 3, 'points not enough'
        assert len(upper_start_points) == len(upper_end_points), 'number of upper start and end points should be equal'
        assert len(lower_start_points) == len(lower_end_points), 'number of upper start and end points should be equal'
        assert h > 0 and w > 0, 'image shape not allowed'

        
        epoch = 30
        
        # make mid_points
        up_mid_point = [((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) for (x0, y0, z0), (x1, y1, z1) in zip(upper_start_points, upper_end_points)]
        down_mid_point = [((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) for (x0, y0, z0), (x1, y1, z1) in zip(lower_start_points, lower_end_points)]
        # TODO: make parabolic: y = a * x^2 + b * x + c

        def ref_curve(a, b, c, x):
            return a * (x ** 2) + b * x + c 

        mat = np.zeros((3,3))
        center_index = int(len(down_mid_point)/2)
        for i in reversed(range(3)):
            mat[0][2-i] = up_mid_point[0][0] ** i
            mat[1][2-i] = down_mid_point[center_index-1][0] ** i
            mat[2][2-i] = up_mid_point[-1][0] ** i
        y=np.array([up_mid_point[0][1], down_mid_point[center_index-1][1], up_mid_point[-1][1]])
        a, b, c = np.linalg.solve(mat, y)

        self.w = w
        solver = CubicSolver(0, w)
        total_center = up_mid_point + down_mid_point
        lr = 1e-12
        # train
        for _ in range(epoch):
            derivative_a = 0.0
            derivative_b = 0.0
            derivative_c = 0.0
            for i in total_center:
                x0 = i[0]
                y0 = i[1]
                coef = [2 * (a**2), 3 * a * b, 2 * a * (c - y0) + b**2 + 1, b * (c - y0) - x0]
                x = np.array(solver(coef[0], coef[1], coef[2], coef[3]))
                if len(x)==0:
                    continue
                else:
                    x = x[((x-x0)**2 + (ref_curve(a, b, c, x)-y0)**2).argmin()]
                    y = ref_curve(a, b, c, x)
                derivative_a += 2 * (y - y0) * (x ** 2)
                derivative_b += 2 * (y - y0) * x
                derivative_c += 2 * (y - y0)
            a = a - lr*derivative_a
            b = b - lr*derivative_b
            c = c - lr*derivative_c

        c += 30
        det = b * b - 4 * a * (c - h)
        
        assert det > 0 , "NO TWO ROOT BY PARABOLIC DET : {}".format(det)
        
        self.left = max((-b - np.sqrt(det)) / (2 * a), 0)
        self.right = min((-b + np.sqrt(det)) / (2 * a), w)

        self.a = a
        self.b = b
        self.c = c
        self.length, _ = integrate.quad(self._prime, self.left, self.right)
    
    def __call__(self, x: float, y: float) -> Union[float, None]:
        """
            find u such that
            x = f(u, v)
            y = g(u, v)
            z = v
        """
        solver = CubicSolver(0, self.w)
        # 2a^2*x^3 + 3*a*b*x^2 + 2*a*c+b^2+1-2*a*y + b*c -b*y-x
        ans = solver(2 * self.a * self.a, 3 * self.a * self.b, 2 * self.a * (self.c - y) + self.b * self.b + 1,
                     self.b * (self.c - y) - x)
        min_dist = 0x7fffffff
        min_x = None
        for x0 in ans:
            y0 = self.a * x0 * x0 + self.b * x0 + self.c
            dist = (x - x0) * (x - x0) + (y - y0) * (y - y0)
            if min_dist > dist:
                min_dist = dist
                min_x = x0
        
        if min_x is None:
            return None

        length, _ = integrate.quad(self._prime, self.left, min_x)
        
        return length

    def _prime(self, t: float) -> float:
        return np.sqrt(1 + (2 * self.a * t + self.b) ** 2)
    
    def get_length(self) -> float:
        return self.length
    
    def get_left_right(self) -> Tuple[float, float]:
        return self.left, self.right
    
    def get_y(self, x: float) -> float:
        return (self.a * x + self.b) * x + self.c