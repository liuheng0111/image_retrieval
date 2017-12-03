# coding=utf-8


import os
import pickle
from PIL import Image
import numpy as np
from tqdm import tqdm
import multiprocessing


"""
检索图片
"""

DATA_ROOT = '/data/wangyuan/koubei_image/data'


def get_num_bits_different(hash1, hash2):
    """
    计算两个哈希值不同比特位的个数.
    """
    return bin(hash1 ^ hash2).count('1')


def query(query_dict, image_dict, method='dhash', top=100):
    """
    搜索相似图片
    :param query_dict: dict, {query_id: encode}
    :param image_dict: dict, {image_id: encode}
    :param method: string, 'dhash', 'vgg19'
    :param top:
    :return:
    """
    # 暴力法
    result = {}

    def compute_similarity(query_id, encoding1, image_id, encoding2, method):
        if method == 'dhash':
            dist = get_num_bits_different(query_encoding, image_encoding)
            if dist <= 50:
                result[query_id].append(image_id)
        elif method == 'vgg19':
            sim = np.dot(query_encoding, image_encoding)
            if sim >= 0.6:
                result[query_id].append(image_id)



    for query_id, query_encoding in tqdm(query_dict.items()):
        if not query_id in result:
            result[query_id] = []
        tmp = []
        p = multiprocessing.Pool(10)
        for image_id, image_encoding in image_dict.items():
            # if method == 'dhash':
            #     sim = 128 - get_num_bits_different(query_encoding, image_encoding)
            # elif method == 'vgg19':
            #     sim = np.dot(query_encoding, image_encoding)
            #
            # if len(tmp) < top:
            #     tmp.append(sim)
            #     result[query_id].append((image_id, sim))
            # else:
            #     if sim <= min(tmp):
            #         continue
            #     else:
            #         idx = np.argmin(tmp)
            #         tmp.remove(tmp[idx])
            #         result[query_id].remove(result[query_id][idx])
            p.apply_async(compute_similarity, args=(query_id, query_encoding, image_id, image_encoding, method, ))
        p.close()
        p.join()

    with open(os.path.join(DATA_ROOT, 'query_result_{}.pkl'.format(method)), 'wb') as f:
        pickle.dump(result, f)
    print('done')
    return result


if __name__ == '__main__':
    with open(os.path.join(DATA_ROOT, 'query_dhash.pkl'), 'rb') as f:
        query_dhash = pickle.load(f)

    with open(os.path.join(DATA_ROOT, 'image_dhash.pkl'), 'rb') as f:
        image_dhash = pickle.load(f)

    qurey_result = query(query_dict=query_dhash, image_dict=image_dhash, method='dhash', top=100)
