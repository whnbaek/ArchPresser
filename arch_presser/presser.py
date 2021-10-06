from __future__ import annotations
from typing import List, Tuple

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from .decode import decode
from .arch import Arch

class ArchPresser:
    def __init__(self, dir_path: str = None, center_points: List[Tuple[int, int]] = None) -> None:
        """
            dir_path: there should be DCM image files in the directory
            center_points: [y0, x0], [y1, x1], ... [yn, xn]
            thickness: pixel distance from arch, pixcels inside the distance is projected to arch
        """
        # parameter check
        if dir_path:
            assert os.path.isdir(dir_path), 'no {} or not a directory'.format(dir_path)
        if center_points:
            for x, y in center_points:
                assert x >= 0 and y >= 0, 'center point coordinate ({}, {}) not allowed'.format(x, y)

        self.dir_path = dir_path
        self.center_points = center_points
        self.data = None
        self.arch = None
        self.panoramic_image = None
    
    def preprocess(self, dir_path: str = None, center_points: List[Tuple[int, int]] = None) -> ArchPresser:
        # parameter check and update
        if dir_path:
            assert os.path.isdir(dir_path), 'no {} or not a directory'.format(dir_path)
            self.dir_path = dir_path
        if center_points:
            for x, y in center_points:
                assert x >= 0 and y >= 0, 'center point coordinate ({}, {}) not allowed'.format(x, y)
            self.center_points = center_points
        assert self.dir_path is not None and self.center_points is not None, 'directory path and center points needed for preprocessing'

        # decode
        print('[decode]')
        self.data = decode(dir_path)
        
        # make arch
        print('[make arch]')
        _, h, w = self.data.shape
        self.arch = Arch(self.center_points, h, w)

        return self
    
    def project(self, thickness: float, resolution: float) -> ArchPresser:
        # parameter check
        assert thickness >= 0, 'thickness {} not allowed'.format(thickness)
        assert resolution >= 0, 'resolution {} not allowed'.format(resolution)
        assert self.data is not None and self.arch is not None, 'data and arch needed for projection'

        # project
        print('[get coordinates and normal vector]')
        d, h, w = self.data.shape
        hp = int(d / resolution)
        wp = int(self.arch.get_length() / resolution)
        
        xp2x_y_n = []
        for xp in tqdm(range(wp)):
            xp2x_y_n.append(self.arch.xp2x_y_n(xp, resolution))

        # ray-sum
        print('[ray-sum]')
        panoramic_image = np.zeros((hp, wp))
        
        for yp in tqdm(range(hp)):
            for xp in tqdm(range(wp), leave = False):
                x, y, n = xp2x_y_n[xp] # n: 2d normal vector
                zt = yp * resolution # yp2z

                z0 = int(zt) # 0 <= z0 < d
                z1 = z0 + 1 # can be z1 >= d
                
                nx, ny = n
                if np.abs(nx) >= np.abs(ny): # iterate x
                    def interpolate_x(t: int) -> float | None:
                        xt = int(x + t)
                        yt = y + (xt - x) * ny / nx
                        if (xt - x) ** 2 + (yt - y) ** 2 > thickness ** 2 or (
                           xt < 0 or xt >= w or yt < 0 or yt >= h):
                            return None
                        
                        # 0 <= xt < w
                        y0 = int(yt) # 0 <= y0 < h 
                        y1 = y0 + 1 # can be y1 >= h

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
                        if res:
                            panoramic_image[yp][xp] += res
                            t += 1
                        else:
                            break
                    t = 0
                    while True:
                        res = interpolate_x(t)
                        if res:
                            panoramic_image[yp][xp] += res
                            t -= 1
                        else:
                            break

                else: # iterate y
                    def interpolate_y(t: int) -> float | None:
                        yt = int(y + t)
                        xt = x + (yt - y) * nx / ny
                        if (xt - x) ** 2 + (yt - y) ** 2 > thickness ** 2 or (
                           xt < 0 or xt >= w or yt < 0 or yt >= h):
                            return None
                        
                        # 0 <= xt < w
                        x0 = int(xt) # 0 <= y0 < h 
                        x1 = x0 + 1 # can be y1 >= h

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
                        if res:
                            panoramic_image[yp][xp] += res
                            t += 1
                        else:
                            break
                    t = 0
                    while True:
                        res = interpolate_y(t)
                        if res:
                            panoramic_image[yp][xp] += res
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