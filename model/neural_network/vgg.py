# coding=utf-8


import numpy as np
from numpy import linalg as LA
import mxnet as mx
from mxnet import gluon, autograd, nd, image
from mxnet.gluon.model_zoo import vision
import cv2
import pickle
from tqdm import tqdm
import scipy as sp
from scipy.spatial.distance import cosine


MODEL_ROOT = '/data/wangyuan/koubei_image/data/models'


def extract_feat(img_path, model):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_shape = (3, 224, 224)
    # model = vision.get_vgg(num_layers=19, pretrained=True, ctx=ctx, root='/data/wangyuan/koubei/models')
    img = cv2.imread(img_path)
    img_224 = ((cv2.resize(img, (224, 224))[:, :, ::-1] / 255.0 - mean) / std).transpose((2, 0, 1))
    img_224 = nd.expand_dims(data=nd.array(img_224), axis=0)

    feat = model.features[0](img_224.as_in_context(ctx))

    for layer in model.features[1:37]:
        feat = layer(feat)

    feat = gluon.nn.GlobalAvgPool2D()(feat)
    # feat = feat[0, :, 0, 0].asnumpy()
    # norm_feat = feat / LA.norm(feat)
    feat = feat[0, :, 0, 0]
    norm_feat = feat / nd.norm(feat)
    return norm_feat.asnumpy()


if __name__ == '__main__':
    with open('/data/wangyuan/koubei_image/data/image_path.pkl', 'rb') as f:
        image_path = pickle.load(f)
    ctx = mx.gpu()
    vgg19 = vision.get_vgg(num_layers=19, pretrained=True, ctx=ctx, root=MODEL_ROOT)

    image_vgg19_pickle_file = '/data/wangyuan/koubei_image/data/image_vgg19.pkl'
    image_vgg19 = {}
    for path in tqdm(image_path):
        idx = path.split('/')[-1].split('.')[0]
        feat = extract_feat(path, vgg19)
        image_vgg19[idx] = feat
    with open(image_vgg19_pickle_file, 'wb') as f:
        pickle.dump(image_vgg19)
    print('done')
