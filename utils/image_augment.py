# coding=utf-8


from PIL import Image
import numpy as np
import os
import pickle
from tqdm import tqdm


"""
图片增强做验证集
随机截取图片，旋转，镜像等
"""


DATA_ROOT = '/data/wangyuan/koubei_image/data'


def image_crop(image, crop):
    x0 = np.random.randint(0, image.size[0] - int(image.size[0] * crop) - 1)
    x1 = x0 + int(image.size[0] * crop)
    y0 = np.random.randint(0, image.size[1] - int(image.size[1] * crop) - 1)
    y1 = y0 + int(image.size[1] * crop)
    image_croped = image.crop((x0, y0, x1, y1))

    return image_croped


def image_augment(image_path, nums=5, crop=0.95, rotate=None):
    """
    生成增强的图片作为测试数据集
    :param image:
    :param nums:
    :param size:
    :return:
    """
    image_labels = {}

    for path in tqdm(image_path):
        image = Image.open(path)
        idx = path.split('/')[-1].split('.')[0]
        # 随机裁剪
        if isinstance(crop, float):
            for i in range(nums):
                image_croped = image_crop(image, crop)
                crop_idx = '{}_crop_{}_{}'.format(idx, int(crop * 100), i)
                # 保存 label 信息：裁剪图片 id -> 原始图片 id
                if crop_idx not in image_labels:
                    image_labels[crop_idx] = []
                image_labels[crop_idx].append(idx)
                path = os.path.join(DATA_ROOT, 'test/{}.jpg'.format(crop_idx))
                image_croped.save(path)
        elif isinstance(crop, list):
            for _crop in crop:
                for i in range(nums):
                    image_croped = image_crop(image, _crop)
                    crop_idx = '{}_crop_{}_{}'.format(idx, int(_crop * 100), i)
                    # 保存 label 信息：裁剪图片 id -> 原始图片 id
                    if crop_idx not in image_labels:
                        image_labels[crop_idx] = []
                    image_labels[crop_idx].append(idx)
                    path = os.path.join(DATA_ROOT, 'test/{}.jpg'.format(crop_idx))
                    image_croped.save(path)

    image_labels_pickle_file = os.path.join(DATA_ROOT, 'image_labels.pkl')
    with open(image_labels_pickle_file, 'wb') as f:
        pickle.dump(image_labels, f)

if __name__ == '__main__':
    image_path_pickle_file = os.path.join(DATA_ROOT, 'image_path.pkl')
    if not os.path.exists(image_path_pickle_file):
        print('image path does not exist.')
    else:
        with open(image_path_pickle_file, 'rb') as f:
            image_path = pickle.load(f)
        random_idx = np.random.randint(0, len(image_path), 1001)
        image_random = [image_path[i] for i, img in enumerate(image_path) if i in random_idx]
        image_augment(image_path=image_random, nums=5, crop=[0.95, 0.85, 0.75], rotate=None)
        print('done.')
