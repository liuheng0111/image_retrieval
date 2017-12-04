# coding=utf-8

import sys
import os
import platform
import numpy as np
from numpy import linalg as LA
import mxnet as mx
from mxnet import gluon, autograd, nd, image
from mxnet.gluon.model_zoo import vision
import cv2
import pickle
from tqdm import tqdm
import multiprocessing

if 'Linux' in platform.platform():
    if not '/data/wangyuan/image_retrieval' in sys.path:
        sys.path.append('//data/wangyuan/image_retrieval')
else:
    if not '/Users/wangyuan_ucas/Desktop/DataScience/image_retrieval' in sys.path:
        sys.path.append('/Users/wangyuan_ucas/Desktop/DataScience/image_retrieval')

from utils.configures import *
from model.utils import try_gpu


def _check_img(path):
    """
    Check if the image path is valid
    :param path: image path
    :return: image if the path is valid else False
    """
    try:
        img = cv2.imread(path)
        if len(img.shape) != 3:
            return False
    except (IOError, AttributeError):
        print('ERROR: invalid image path: {}'.format(path))
        return False
    return True


def transform(data):
    mean = nd.array([0.485, 0.456, 0.406])
    std = nd.array([0.229, 0.224, 0.225])
    data = ((mx.image.imresize(data.astype('float32'), 224, 224) / 255.0 - mean) / std).transpose((2, 0, 1))
    data = nd.expand_dims(data, axis=0)
    return data


def read_image(dir_path):

    path_list = [path for path in os.listdir(dir_path)]
    path_list = [os.path.join(dir_path, path) for path in path_list]

    p = multiprocessing.Pool(NUM_THREADS)
    print('Check if the path is valid...')
    check_result = []
    for path in tqdm(path_list):
        check_result.append(p.apply_async(_check_img, args=(path, )))
    p.close()
    p.join()

    valid_path_list = []
    ids = []
    for path, check in tqdm(zip(path_list, check_result)):
        if check.get():
            valid_path_list.append(path)
            ids.append(path.split('/')[-1].split('.')[0])
        else:
            continue
    print('Reading images...')
    n = len(valid_path_list)
    num_batches = n // BATCH_SIZE
    ids = nd.array(ids)
    for i in range(num_batches + 1):
        _path_list = valid_path_list[i * BATCH_SIZE:min((i + 1) * BATCH_SIZE, n)]
        _ids = ids[i * BATCH_SIZE:min((i + 1) * BATCH_SIZE, n)]
        _data = []
        for path in _path_list:
            img = mx.image.imread(path)
            _data.append(img)
        _data = [img for img in map(transform, _data)]
        _data = nd.concatenate(_data)
        yield _data, _ids.astype('int')


def extract_feat(ids, data, model):
    ctx = try_gpu()
    if len(ctx) > 1:
        _data_list = gluon.utils.split_and_load(data=data, ctx_list=ctx)
        _id_list = gluon.utils.split_and_load(data=ids, ctx_list=ctx)
    else:
        _data_list = [data.as_in_context(ctx[0])]
        _id_list = [ids.as_in_context(ctx[0])]
    # print(_id_list)
    data_list = []
    id_list = []

    for _ids, _data in zip(_id_list, _data_list):
        feats = model(_data)
        feats = map(lambda x: x / nd.norm(x), feats[:, :, 0, 0])
        feats = [v.expand_dims(axis=0) for v in feats]
        feats = nd.concatenate(feats)
        id_list.append(_ids)
        data_list.append(feats)
    id_list = nd.concatenate(id_list)
    data_list = nd.concatenate(data_list)
    return id_list, data_list


if __name__ == '__main__':
    # TODO: delete
    # with open('/data/wangyuan/koubei_image/data/image_path.pkl', 'rb') as f:
    #     image_path = pickle.load(f)

    # ctx = try_gpu()
    # vgg19 = vision.get_vgg(num_layers=19, pretrained=True, ctx=ctx, root=MODEL_ROOT)
    #
    # image_vgg19_pickle_file = '/data/wangyuan/koubei_image/data/image_vgg19.pkl'
    # image_vgg19 = {}
    # for path in tqdm(image_path):
    #     idx = path.split('/')[-1].split('.')[0]
    #     feat = extract_feat(path, vgg19)
    #     image_vgg19[idx] = feat
    # with open(image_vgg19_pickle_file, 'wb') as f:
    #     pickle.dump(image_vgg19)

    ctx = try_gpu()
    vgg19 = vision.get_vgg(num_layers=19, pretrained=True, ctx=ctx, root=MODEL_ROOT)
    model = gluon.nn.Sequential()
    with model.name_scope():
        for layer in vgg19.features[:37]:
            model.add(layer)
        model.add(gluon.nn.GlobalAvgPool2D())

    dir_path = os.path.join(DATA_ROOT, 'train')

    id_list = []
    feat_list = []
    for data, ids in tqdm(read_image(dir_path)):
        _ids, _feat = extract_feat(ids, data, model)
        id_list.append(_ids.asnumpy())
        feat_list.append(_feat.asnumpy())

    print('Saving VGG19 features...')
    id_list = np.concatenate(id_list)
    feat_list = np.concatenate(feat_list)
    image_vgg19 = {}
    for idx, feat in tqdm(zip(id_list, feat_list)):
        image_vgg19[idx] = feat
    with open(IMAGE_VGG19, 'wb') as f:
        pickle.dump(image_vgg19, f)
    print('done')
