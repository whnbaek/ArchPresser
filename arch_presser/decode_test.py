from decode import decode
from PIL import Image
import numpy as np

def main():
    image = decode("../CT_case2/18")

    print(image.shape)

    image = (np.maximum(image[160] - 500, 0) * 255 / 500).astype(np.uint8)
    image = Image.fromarray(image, 'L')
    image.save('decode_test.png')

if __name__ == '__main__':
    main()