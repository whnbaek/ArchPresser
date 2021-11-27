import os
import pickle
from arch_presser import ArchPresser

def main():
    dir_path = './datasets/18'
    upper_start_points = [(19, 503, 80), (146, 309, 110), (172, 228, 100),
                          (182, 201,  90), (205, 177,  80), (238, 159, 100), (263, 141, 100),
                          (304, 142, 100), (330, 153, 110), (363, 174,  80), (384, 190, 110),
                          (394, 231, 110), (404, 347, 110), (532, 510, 80)]
    lower_start_points = [(19, 503, 270), (120, 364, 260), (147, 257, 300),
                          (170, 217, 310), (190, 187, 320), (216, 168, 320), (244, 150, 300),
                          (266, 144, 300), (290, 145, 300), (310, 150, 300), (336, 177, 330),
                          (363, 191, 320), (387, 227, 320), (409, 258, 300), (420, 309, 290),
                          (532, 510, 270)]
    upper_end_points = [(19, 503, 200), (140, 300, 190), (167, 216, 180),
                        (180, 183, 180), (191, 140, 190), (221, 107, 190), (261,  91, 190),
                        (304,  91, 190), (345, 107, 190), (375, 137, 195), (385, 180, 190),
                        (396, 216, 190), (416, 339, 180), (532, 510, 200)]
    lower_end_points = [(19, 503, 200), (137, 328, 220), (158, 247, 220),
                        (175, 203, 220), (187, 168, 230), (201, 127, 220), (233, 103, 220),
                        (262,  95, 225), (290, 95, 225), (324, 103, 225), (353, 123, 225),
                        (371, 159, 225), (378, 197, 225), (394, 239, 230), (410, 292, 210),
                        (532, 510, 200)]
    thickness = 50.0
    pixel_size = 1.0

    pkl_path = './pickle/{:.1f}_{:.1f}.pkl'.format(thickness, pixel_size)
    if os.path.isfile(pkl_path):
        with open(pkl_path, 'rb') as f:
            presser = pickle.load(f)
    else:
        presser = (ArchPresser().preprocess(dir_path = dir_path,
                                            upper_start_points = upper_start_points,
                                            upper_end_points = upper_end_points,
                                            lower_start_points = lower_start_points,
                                            lower_end_points = lower_end_points,
                                            pixel_size = pixel_size
                                            )
                                .project(thickness = thickness))
        with open(pkl_path, 'wb') as f:
            pickle.dump(presser, f)

    image_path = './results/v2_{:.1f}_{:.1f}.png'.format(thickness, pixel_size)
    _ = presser.press(image_path = image_path, clamp = (0.0, 260000.0))

if __name__ == '__main__':
    main()