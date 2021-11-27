from __future__ import annotations
from typing import List, Tuple

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import interpolate

from .decode import decode
from .arch import Arch

class ArchPresser:
    def __init__(self, dir_path: str = None, upper_start_points: List[Tuple[float, float, float]] = None,
                 lower_start_points: List[Tuple[float, float, float]] = None, 
                 upper_end_points: List[Tuple[float, float, float]] = None,
                 lower_end_points: List[Tuple[float, float, float]] = None,
                 pixel_size: float = None) -> None:
        """
            dir_path: there should be DCM image files in the directory
            center_points: [y0, x0], [y1, x1], ... [yn, xn]
            thickness: pixel distance from arch, pixcels inside the distance is projected to arch
        """
        # parameter check
        if dir_path is not None:
            assert os.path.isdir(dir_path), 'no {} or not a directory'.format(dir_path)
        if upper_start_points is not None:
            for x, y, z in upper_start_points:
                assert x >= 0 and y >= 0 and z >= 0, 'upper start point coordinate ({}, {}, {}) not allowed'.format(x, y, z)
        if lower_start_points is not None:
            for x, y, z in lower_start_points:
                assert x >= 0 and y >= 0 and z >= 0, 'lower start point coordinate ({}, {}, {}) not allowed'.format(x, y, z)
        if upper_end_points is not None:
            for x, y, z in upper_end_points:
                assert x >= 0 and y >= 0 and z >= 0, 'upper end point coordinate ({}, {}, {}) not allowed'.format(x, y, z)
        if lower_end_points is not None:
            for x, y, z in lower_end_points:
                assert x >= 0 and y >= 0 and z >= 0, 'lower end point coordinate ({}, {}, {}) not allowed'.format(x, y, z)
        if pixel_size is not None:
            assert pixel_size > 0, 'pixel size not allowed'

        self.dir_path = dir_path
        self.upper_start_points = upper_start_points
        self.upper_end_points = upper_end_points
        self.lower_start_points = lower_start_points
        self.lower_end_points = lower_end_points
        self.pixel_size = pixel_size
        self.data = None
        self.arch = None
        self.panoramic_image = None
    
    def preprocess(self, dir_path: str = None, upper_start_points: List[Tuple[float, float, float]] = None,
                   upper_end_points: List[Tuple[float, float, float]] = None, 
                   lower_start_points: List[Tuple[float, float, float]] = None,
                   lower_end_points: List[Tuple[float, float, float]] = None,
                   pixel_size: float = None) -> ArchPresser:
        # parameter check and update
        if dir_path is not None:
            assert os.path.isdir(dir_path), 'no {} or not a directory'.format(dir_path)
            self.dir_path = dir_path
        if upper_start_points is not None:
            for x, y, z in upper_start_points:
                assert x >= 0 and y >= 0 and z >= 0, 'upper_start point coordinate ({}, {}, {}) not allowed'.format(x, y, z)
            self.upper_start_points = upper_start_points
        if upper_end_points is not None:
            for x, y, z in upper_end_points:
                assert x >= 0 and y >= 0 and z >= 0, 'upper_end point coordinate ({}, {}, {}) not allowed'.format(x, y, z)
            self.upper_end_points = upper_end_points
        if lower_start_points is not None:
            for x, y, z in lower_start_points:
                assert x >= 0 and y >= 0 and z >= 0, 'lower_start point coordinate ({}, {}, {}) not allowed'.format(x, y, z)
            self.lower_start_points = lower_start_points
        if lower_end_points is not None:
            for x, y, z in lower_end_points:
                assert x >= 0 and y >= 0 and z >= 0, 'lower_end point coordinate ({}, {}, {}) not allowed'.format(x, y, z)
            self.lower_end_points = lower_end_points
        if pixel_size is not None:
            assert pixel_size > 0, 'pixel size not allowed'
            self.pixel_size = pixel_size
        assert (self.dir_path is not None and self.pixel_size is not None and
                self.upper_start_points is not None and self.lower_start_points and
                self.upper_end_points is not None and self.lower_end_points is not None), 'directory path and center points needed for preprocessing'

        # decode
        print('[decode]')
        self.data = decode(dir_path)
        
        # make arch
        print('[make arch]')
        d, h, w = self.data.shape
        self.arch = Arch(self.upper_start_points, self.upper_end_points, self.lower_start_points,
                         self.lower_end_points, h, w)
        self.hp = int(d / self.pixel_size)
        self.wp = int(self.arch.get_length() / self.pixel_size)

        points = []
        for (x0, y0, z0), (x1, y1, z1) in zip(self.upper_start_points, self.upper_end_points):
            for z in range(z0):
                points.append((x0, y0, z))
            for z in range(z0, z1):
                x = x0 + (z - z0) * (x1 - x0) / (z1 - z0)
                y = y0 + (z - z0) * (y1 - y0) / (z1 - z0)
                points.append((x, y, z))
        for (x0, y0, z0), (x1, y1, z1) in zip(self.lower_end_points, self.lower_start_points):
            for z in range(z0, z1):
                x = x0 + (z - z0) * (x1 - x0) / (z1 - z0)
                y = y0 + (z - z0) * (y1 - y0) / (z1 - z0)
                points.append((x, y, z))
            for z in range(z1, d):
                points.append((x1, y1, z))
        us = []
        vs = []
        xs = []
        ys = []
        for x, y, z in points:
            l = self.arch(x, y)
            if l is not None:
                us.append(l / self.pixel_size)
                vs.append(z / self.pixel_size)
                xs.append(x)
                ys.append(y)
        self.x_tck = interpolate.bisplrep(us, vs, xs)
        self.y_tck = interpolate.bisplrep(us, vs, ys)
        
        return self
    
    def project(self, thickness: float) -> ArchPresser:
        # parameter check
        assert thickness >= 0, 'thickness {} not allowed'.format(thickness)
        assert self.data is not None and self.arch is not None, 'data and arch needed for projection'

        # project
        print('[get coordinates and normal vector]')
        d, h, w = self.data.shape
        
        xs = interpolate.bisplev(range(self.wp), range(self.hp), self.x_tck)
        ys = interpolate.bisplev(range(self.wp), range(self.hp), self.y_tck)
        dxdus = interpolate.bisplev(range(self.wp), range(self.hp), self.x_tck, dx = 1)
        dxdvs = interpolate.bisplev(range(self.wp), range(self.hp), self.x_tck, dy = 1)
        dydus = interpolate.bisplev(range(self.wp), range(self.hp), self.y_tck, dx = 1)
        dydvs = interpolate.bisplev(range(self.wp), range(self.hp), self.y_tck, dy = 1)
        
        # ray-sum
        print('[ray-sum]')
        panoramic_image = np.zeros((self.hp, self.wp))
        
        for v in tqdm(range(self.hp)):
            for u in range(self.wp):
                x = xs[u, v]
                y = ys[u, v]
                z = v * self.pixel_size
                nx = dydus[u, v] * self.pixel_size
                ny = -dxdus[u, v] * self.pixel_size
                nz = dxdus[u, v] * dydvs[u, v] - dydus[u, v] * dxdvs[u, v]
                size = np.sqrt(nx * nx + ny * ny + nz * nz)
                nx /= size
                ny /= size
                nz /= size
                
                if np.abs(nx) >= np.abs(ny) and np.abs(nx) >= np.abs(nz): # iterate x
                    def interpolate_x(t: int) -> float | None:
                        xt = int(x + t)
                        yt = y + (xt - x) * ny / nx
                        zt = z + (xt - x) * nz / nx
                        if (xt - x) ** 2 + (yt - y) ** 2 + (zt - z) ** 2 > thickness ** 2 or (
                           xt < 0 or xt >= w or yt < 0 or yt >= h or zt < 0 or zt >= d):
                            return None
                        
                        # 0 <= xt < w
                        y0 = int(yt) # 0 <= y0 < h 
                        y1 = y0 + 1 # can be y1 >= h
                        z0 = int(zt) # 0 <= z0 < d
                        z1 = z0 + 1 # can be z1 >= d

                        res = self.data[z0][y0][xt] * (z1 - zt) * (y1 - yt)
                        if z1 < d:
                            res += self.data[z1][y0][xt] * (zt - z0) * (y1 - yt)
                        if y1 < h:
                            res += self.data[z0][y1][xt] * (z1 - zt) * (yt - y0)
                        if z1 < d and y1 < h:
                            res += self.data[z1][y1][xt] * (zt - z0) * (yt - y0)
                        return res
                        
                    t = 1
                    while True:
                        res = interpolate_x(t)
                        if res is not None:
                            panoramic_image[v][u] += res
                            t += 1
                        else:
                            break
                    t = 0
                    while True:
                        res = interpolate_x(t)
                        if res is not None:
                            panoramic_image[v][u] += res
                            t -= 1
                        else:
                            break

                elif np.abs(ny) >= np.abs(nx) and np.abs(ny) >= np.abs(nz): # iterate y
                    def interpolate_y(t: int) -> float | None:
                        yt = int(y + t)
                        xt = x + (yt - y) * nx / ny
                        zt = z + (yt - y) * nz / ny
                        if (xt - x) ** 2 + (yt - y) ** 2 + (zt - z) ** 2 > thickness ** 2 or (
                           xt < 0 or xt >= w or yt < 0 or yt >= h or zt < 0 or zt >= d):
                            return None
                        
                        # 0 <= xt < w
                        x0 = int(xt) # 0 <= y0 < h 
                        x1 = x0 + 1 # can be y1 >= h
                        z0 = int(zt) # 0 <= z0 < d
                        z1 = z0 + 1 # can be z1 >= d

                        res = self.data[z0][yt][x0] * (z1 - zt) * (x1 - xt)
                        if z1 < d:
                            res += self.data[z1][yt][x0] * (zt - z0) * (x1 - xt)
                        if x1 < w:
                            res += self.data[z0][yt][x1] * (z1 - zt) * (xt - x0)
                        if z1 < d and x1 < w:
                            res += self.data[z1][yt][x1] * (zt - z0) * (xt - x0)
                        return res
                        
                    t = 1
                    while True:
                        res = interpolate_y(t)
                        if res is not None:
                            panoramic_image[v][u] += res
                            t += 1
                        else:
                            break
                    t = 0
                    while True:
                        res = interpolate_y(t)
                        if res is not None:
                            panoramic_image[v][u] += res
                            t -= 1
                        else:
                            break
                else: #iterate z
                    def interpolate_z(t: int) -> float | None:
                        zt = int(z + t)
                        xt = x + (zt - z) * nx / nz
                        yt = y + (zt - z) * ny / nz
                        if (xt - x) ** 2 + (yt - y) ** 2 + (zt - z) ** 2 > thickness ** 2 or (
                           xt < 0 or xt >= w or yt < 0 or yt >= h or zt < 0 or zt >= d):
                            return None
                        
                        # 0 <= xt < w
                        x0 = int(xt) # 0 <= x0 < w 
                        x1 = x0 + 1 # can be x1 >= w
                        y0 = int(yt) # 0 <= y0 < h 
                        y1 = y0 + 1 # can be y1 >= h

                        res = self.data[zt][y0][x0] * (y1 - yt) * (x1 - xt)
                        if y1 < h:
                            res += self.data[zt][y1][x0] * (yt - y0) * (x1 - xt)
                        if x1 < w:
                            res += self.data[zt][y0][x1] * (y1 - yt) * (xt - x0)
                        if y1 < h and x1 < w:
                            res += self.data[zt][y1][x1] * (yt - y0) * (xt - x0)
                        return res
                        
                    t = 1
                    while True:
                        res = interpolate_z(t)
                        if res is not None:
                            panoramic_image[v][u] += res
                            t += 1
                        else:
                            break
                    t = 0
                    while True:
                        res = interpolate_z(t)
                        if res is not None:
                            panoramic_image[v][u] += res
                            t -= 1
                        else:
                            break

        self.panoramic_image = panoramic_image

        return self

    def press(self, image_path: str, clamp: Tuple[float, float]) -> ArchPresser:
        # parameter check
        assert self.panoramic_image is not None, 'panoramic image needed for pressing'

        # normalize
        image = self.panoramic_image
        image = np.clip(image, clamp[0], clamp[1])
        image = (image - image.min()) * 255.0 / (image.max() - image.min())
        image = image.astype(np.uint8)
        
        # press image
        Image.fromarray(image, 'L').save(image_path)

        return self
