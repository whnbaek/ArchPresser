from arch import Arch

def main():
    center_points = [(143, 304), (168, 219), (179, 186), (198, 151), (227, 123), (262, 108),
                     (304, 108), (339, 124), (368, 148), (384, 184), (396, 220), (413, 340)]
    resolution = 1.0

    arch = Arch(center_points = center_points, h = 550, w = 550)
    
    length = arch.get_length()
    print('length: {:.1f}'.format(length))

    wp = int(length / resolution)
    for xp in range(0, wp, 25):
        x, y, n = arch.xp2x_y_n(xp, resolution)
        nx, ny = n
        print('xp: {:.1f}, x: {:.1f}, y: {:.1f}, n: ({:.1f}, {:.1f})'.format(xp, x, y, nx, ny))

if __name__ == '__main__':
    main()