# coding=utf-8


import os
from PIL import Image
import pickle
from tqdm import tqdm
# from .configures import *


DATA_ROOT = '/data/wangyuan/koubei_image/data'


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


if __name__ == '__main__':
    img_paths = get_image_path(DATA_ROOT)
