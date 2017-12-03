# coding=utf-8


from PIL import Image
from utils import *


def get_image_pixl(img_path):
    img = Image.open(img_path)
    return img.size[0] * img.size[1]


if __name__ == '__main__':
    img_paths = get_image_path(IMAGE_ROOT)
    print(get_image_pixl(img_paths[0]))
