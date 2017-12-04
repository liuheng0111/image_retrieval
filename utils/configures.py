# coding=utf-8

import platform
import os


if 'Linux' in platform.platform():
    DATA_ROOT = '/data/wangyuan/image_retrieval/data'
    MODULE_ROOT = '/data/wangyuan/image_retrieval'
    NUM_THREADS = 10
    BATCH_SIZE = 256
else:
    DATA_ROOT = '/Users/wangyuan_ucas/Desktop/DataScience/image_retrieval/data'
    MODULE_ROOT = '/Users/wangyuan_ucas/Desktop/DataScience/image_retrieval'
    NUM_THREADS = 2
    BATCH_SIZE = 5

IMAGE_DHASH = os.path.join(DATA_ROOT, 'image_dhash.pkl')
QUERY_DHASH = os.path.join(DATA_ROOT, 'query_dhash.pkl')
QUERY_RESULT_DHASH = os.path.join(DATA_ROOT, 'query_result_dhash.pkl')
VGG19 = os.path.join(DATA_ROOT, 'models/vgg19-f7134366.params')
IMAGE_VGG19 = os.path.join(DATA_ROOT, 'image_vgg19.pkl')
MODEL_ROOT = os.path.join(DATA_ROOT, 'models')

