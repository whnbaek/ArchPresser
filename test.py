import os
import pickle
from arch_presser import ArchPresser

def main():
    dir_path = './datasets/18'
    center_points = [(143, 304), (168, 219), (179, 186), (198, 151), (227, 123), (262, 108),
                     (304, 108), (339, 124), (368, 148), (384, 184), (396, 220), (413, 340)]
    thickness = 50.0
    resolution = 1.0

    pkl_path = './pickle/{:.1f}_{:.1f}_v1.pkl'.format(thickness, resolution)
    if os.path.isfile(pkl_path):
        with open(pkl_path, 'rb') as f:
            presser = pickle.load(f)
    else:
        presser = (ArchPresser().preprocess(dir_path = dir_path, center_points = center_points)
                                .project(thickness = thickness, resolution = resolution))
        with open(pkl_path, 'wb') as f:
            pickle.dump(presser, f)

    image_path = './results/{:.1f}_{:.1f}_{:.1f}_v1.png'.format(thickness, resolution, 150000.0)
    _ = presser.press(image_path = image_path, clamp = (0.0, 150000.0))

if __name__ == '__main__':
    main()