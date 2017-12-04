# coding=utf-8


import os, sys
from PIL import Image
import pickle
from tqdm import tqdm
import numpy as np
import platform


if 'Linux' in platform.platform():
    if not '/data/wangyuan/image_retrieval' in sys.path:
        sys.path.append('/data/wangyuan/image_retrieval')
else:
    if not '/Users/wangyuan_ucas/Desktop/DataScience/image_retrieval' in sys.path:
        sys.path.append('/Users/wangyuan_ucas/Desktop/DataScience/image_retrieval')

from utils.configures import *


# TODO: delete the following codes
def get_image_path(dir_path, istrain=True, force=False):
    """
    获取有效图片的路径列表
    :param dir_path:
    :return:
    """
    if istrain:
        dir_path = os.path.join(DATA_ROOT, 'train')
        print('训练数据：{}'.format(dir_path))
    else:
        dir_path = os.path.join(DATA_ROOT, 'test')
        print('测试数据：{}'.format(dir_path))

    image_path_pickle_file = os.path.join(DATA_ROOT, 'image_path.pkl')
    # 已存在
    if os.path.exists(image_path_pickle_file) and not force:
        print('路径文件已存在，直接读取...')
        with open(image_path_pickle_file, 'rb') as f:
            image_path = pickle.load(f)
        return image_path

    image_path = os.listdir(dir_path)
    image_path = [os.path.join(dir_path, img) for img in image_path]
    print('Number of images: {}'.format(len(image_path)))
    # 获取有效的图片路径
    print('获取有效的图片路径...')
    valid_img_path = []
    for img in tqdm(image_path):
        try:
            Image.open(img)
        except OSError:
            print('invalid image: {}'.format(img))
            continue
        valid_img_path.append(img)
    print('Number of valid images: {}'.format(len(valid_img_path)))
    with open(image_path_pickle_file, 'wb') as f:
        pickle.dump(valid_img_path, f)
    return valid_img_path


def _check_img(path):
    """
    Check if the image path is valid
    :param path: image path
    :return: image if the path is valid else False
    """
    try:
        Image.open(path)
    except IOError:
        print('ERROR: invalid image path: {}'.format(path))
        return False
    return True


def read_image(dir_path):
    """
    read image
    :param dir_path: directory path containing images
    :return: image id and image
    """
    path_list = [path for path in os.listdir(dir_path) if not path.startswith('.')]
    id_list = [path.split('.')[0] for path in path_list]
    path_list = [os.path.join(dir_path, path) for path in path_list]
    for idx, path in zip(id_list, path_list):
        if _check_img(path):
            yield idx, Image.open(path)


if __name__ == '__main__':
    dir_path = os.path.join(DATA_ROOT, 'train')
    for idx, image in read_image(dir_path):
        print(idx, np.array(image).shape)
