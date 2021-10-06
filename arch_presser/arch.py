from typing import List, Tuple
import numpy as np
import scipy.interpolate as interpolate
import scipy.integrate as integrate

class Arch:
    epsilon = 1e-3

    def __init__(self, center_points: List[Tuple[int, int]], h: int, w: int) -> None:
        assert len(center_points) >= 2, 'center points not enough'
        assert h > 0 and w > 0, 'image shape not allowed'

        # center point sorting, x first
        center_points.sort()

        # add start and end points
        x0, y0 = center_points[0]
        x1, y1 = center_points[1]
        x = max(x0 + (h - y0) * (x1 - x0) / (y1 - y0), 0)
        y = (x - x0) * (y1 - y0) / (x1 - x0) + y0
        center_points.insert(0, (x, y))

        x0, y0 = center_points[-2]
        x1, y1 = center_points[-1]
        x = min(x0 + (h - y0) * (x1 - x0) / (y1 - y0), w)
        y = (x - x0) * (y1 - y0) / (x1 - x0) + y0
        center_points.append((x, y))

        # make spline and its derivative
        self.spline = interpolate.CubicSpline([x for x, _ in center_points],
                                              [y for _, y in center_points],
                                              bc_type = 'natural')
        self.spline_prime = self.spline.derivative()

        # find x-coordinate of both side end points
        self.ox = center_points[0][0]
        ex = center_points[-1][0]

        # save past execution information for fast xp2x_y_n
        self.past_x = self.ox
        self.past_xp_res = 0

        # set length
        self.length, _ = integrate.quad(self.length_prime, self.ox, ex, epsabs = self.epsilon)

    def xp2x_y_n(self, xp: int, resolution: float) -> Tuple[float, float, Tuple[float, float]]:
        # find x such that length of spline from ox to x equals xp * resolution
        xp_res = xp * resolution
        
        left = min(self.past_x, self.past_x + xp_res - self.past_xp_res)
        right = max(self.past_x, self.past_x + xp_res - self.past_xp_res)

        while right - left >= self.epsilon:
            mid = (left + right) / 2
            length, _ = integrate.quad(self.length_prime, self.ox, mid, epsabs = self.epsilon)
            if length < xp_res:
                left = mid
            else:
                right = mid
        
        x = (left + right) / 2

        # update past execution information
        self.past_x = x
        self.past_xp_res = xp_res

        # get y and n
        y = self.spline(x)
        y_prime = self.spline_prime(x)
        n = (y_prime, -1)

        return x, y, n
    
    def get_length(self) -> float:
        return self.length

    # make length derivative function
    def length_prime(self, t: float) -> float:
        return np.sqrt(1 + self.spline_prime(t) ** 2)